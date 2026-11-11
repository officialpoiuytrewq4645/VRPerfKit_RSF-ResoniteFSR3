#include "openvr_hooks.h"
#include "hooks.h"
#include "logging.h"
#include "win_header_sane.h"
#include "openvr.h"
#include "openvr_manager.h"
#include "resolution_scaling.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

// Patched hook layer (Resonite / Unity 2019 fix):
//  * The original code hooked only the FIRST IVRCompositor_xxx / IVRSystem_xxx interface that was
//    requested and ignored "FnTable:" interfaces entirely. Unity's built-in OpenVR support may submit
//    through a different interface version (or a C function table), so its frames were never seen.
//  * The original code also assumed Submit is always vtable slot 5, which is wrong for
//    IVRCompositor_029 (GetSubmitTexture was inserted at slot 5, Submit moved to slot 6).
// This version hooks every requested compositor/system interface (C++ vtable and FnTable variants),
// uses per-version method slots, de-duplicates identical targets, and uses per-thread re-entrancy
// guards so a frame is post-processed exactly once even when one hooked layer forwards to another.

namespace vrperfkit {
	extern HMODULE g_moduleSelf;

	namespace {
		void *g_clientCoreInstance = nullptr;

		std::recursive_mutex g_interfaceHookMutex;
		std::vector<void *> g_slotHookTargets;

		// ---------------------------------------------------------------------------------------
		// Re-entrancy guards: only the outermost hooked call on a thread does the mod's work.
		// ---------------------------------------------------------------------------------------
		thread_local int t_submitDepth = 0;
		thread_local int t_waitGetPosesDepth = 0;
		thread_local int t_postPresentHandoffDepth = 0;
		thread_local int t_renderTargetSizeDepth = 0;

		struct DepthGuard {
			int &depth;
			explicit DepthGuard(int &depthCounter) : depth(depthCounter) { ++depth; }
			~DepthGuard() { --depth; }
			bool IsOutermost() const { return depth == 1; }
		};

		void LogFirstCall(std::atomic<bool> &alreadyLogged, const char *hookKind, int slotIndex) {
			bool expected = false;
			if (alreadyLogged.compare_exchange_strong(expected, true)) {
				LOG_INFO << "First call received through " << hookKind << " hook slot " << slotIndex;
			}
		}

		// ---------------------------------------------------------------------------------------
		// ResoniteFSR3 integration: the FSR 3.1 renderer plugin renders at a reduced Unity
		// XRSettings.renderViewportScale and upscales back to the full eye inside the same texture.
		// Unity still submits the reduced viewport as the texture bounds, which would make the compositor
		// crop/stretch the corner. When a scale is set, expand the submitted bounds back to the full eye:
		// keep each rect's minimum corner (where Unity anchors the scaled viewport), divide the size by
		// the scale, and preserve the original orientation (flips).
		// ---------------------------------------------------------------------------------------
		std::atomic<float> g_submitViewportScale { 1.0f };
		float g_lastLoggedScale[2] = { -1.0f, -1.0f };

		const vr::VRTextureBounds_t *ExpandBoundsForViewportScale(vr::EVREye eye, const vr::VRTextureBounds_t *bounds, vr::VRTextureBounds_t &storage) {
			float scale = g_submitViewportScale.load();
			if (bounds == nullptr || scale >= 0.999f || scale <= 0.05f) {
				return bounds;
			}

			auto assign = [](float first, float second, float lower, float upper, float &outFirst, float &outSecond) {
				if (first <= second) { outFirst = lower; outSecond = upper; }
				else { outFirst = upper; outSecond = lower; }
			};

			// Horizontal: Unity scales the whole double-wide viewport, packing the scaled eyes side by side
			// (left at [0, s/2], right at [s/2, s]), so the right eye's region does NOT start at 0.5.
			// Map each eye back to its true half of the double-wide texture.
			float uSpanFull = std::abs(bounds->uMax - bounds->uMin) / scale;
			if (uSpanFull < 0.75f) {
				float lower = eye == vr::Eye_Right ? 0.5f : 0.0f;
				assign(bounds->uMin, bounds->uMax, lower, lower + 0.5f, storage.uMin, storage.uMax);
			}
			else {
				float lower = (std::min)(bounds->uMin, bounds->uMax);
				assign(bounds->uMin, bounds->uMax, lower, (std::min)(lower + uSpanFull, 1.0f), storage.uMin, storage.uMax);
			}

			// Vertical: the scaled viewport is anchored at the minimum corner
			float vLower = (std::min)(bounds->vMin, bounds->vMax);
			float vSpanFull = std::abs(bounds->vMax - bounds->vMin) / scale;
			assign(bounds->vMin, bounds->vMax, vLower, (std::min)(vLower + vSpanFull, 1.0f), storage.vMin, storage.vMax);

			int eyeIndex = eye == vr::Eye_Right ? 1 : 0;
			if (g_lastLoggedScale[eyeIndex] != scale) {
				g_lastLoggedScale[eyeIndex] = scale;
				LOG_INFO << "ResoniteFSR3 viewport scale " << scale << " eye " << eyeIndex
					<< ": bounds u(" << bounds->uMin << "," << bounds->uMax << ") v(" << bounds->vMin << "," << bounds->vMax
					<< ") -> u(" << storage.uMin << "," << storage.uMax << ") v(" << storage.vMin << "," << storage.vMax << ")";
			}
			return &storage;
		}

		// ---------------------------------------------------------------------------------------
		// ResoniteFSR3 integration: FSR 3.1 upscaling. The renderer plugin renders the eyes at reduced
		// resolution, upscales each eye with FSR 3.1 into its own full-resolution texture, and registers
		// those textures here. At submit, the eye's texture is swapped for the upscaled one (whole texture,
		// linear color space). Depth/pose submit flags are dropped because they describe the original
		// texture's struct layout and contents.
		// ---------------------------------------------------------------------------------------
		std::atomic<void *> g_submitTextureOverride[2] = { nullptr, nullptr };
		std::atomic<int> g_submitOverrideFlipVertical { 0 };
		bool g_loggedOverride[2] = { false, false };

		void ApplySubmitTextureOverride(vr::EVREye eye, OpenVrSubmitInfo &info, vr::Texture_t &textureStorage, vr::VRTextureBounds_t &boundsStorage) {
			int eyeIndex = eye == vr::Eye_Right ? 1 : 0;
			void *overrideHandle = g_submitTextureOverride[eyeIndex].load();
			if (overrideHandle == nullptr || info.texture == nullptr) {
				g_loggedOverride[eyeIndex] = false;
				return;
			}

			textureStorage.handle = overrideHandle;
			textureStorage.eType = vr::TextureType_DirectX;
			textureStorage.eColorSpace = vr::ColorSpace_Linear;
			// The FSR output follows Unity's bottom-up render texture layout; flipping V presents it upright
			bool flipVertical = g_submitOverrideFlipVertical.load() != 0;
			boundsStorage = { 0.0f, flipVertical ? 1.0f : 0.0f, 1.0f, flipVertical ? 0.0f : 1.0f };

			if (!g_loggedOverride[eyeIndex]) {
				g_loggedOverride[eyeIndex] = true;
				LOG_INFO << "ResoniteFSR3 submit override eye " << eyeIndex << ": texture " << overrideHandle
					<< " replaces " << info.texture->handle << " (flags " << info.submitFlags << " -> 0, flipV " << flipVertical << ")";
			}

			info.texture = &textureStorage;
			info.bounds = &boundsStorage;
			info.submitFlags = vr::Submit_Default;
		}

		// ---------------------------------------------------------------------------------------
		// Shared work for each hooked method.
		// ---------------------------------------------------------------------------------------
		template<typename CallOriginal>
		vr::EVRCompositorError ProcessSubmit(vr::EVREye eye, const vr::Texture_t *texture, const vr::VRTextureBounds_t *bounds,
				vr::EVRSubmitFlags submitFlags, CallOriginal callOriginal) {
			DepthGuard guard(t_submitDepth);
			if (!guard.IsOutermost()) {
				return callOriginal(eye, texture, bounds, submitFlags);
			}

			OpenVrSubmitInfo info { eye, texture, bounds, submitFlags };
			g_openVr.OnSubmit(info);
			vr::VRTextureBounds_t expandedBounds;
			info.bounds = ExpandBoundsForViewportScale(eye, info.bounds, expandedBounds);
			vr::Texture_t overrideTexture;
			vr::VRTextureBounds_t overrideBounds;
			ApplySubmitTextureOverride(eye, info, overrideTexture, overrideBounds);
			g_openVr.PreCompositorWorkCall(true);
			auto error = callOriginal(info.eye, info.texture, info.bounds, info.submitFlags);
			if (error != vr::VRCompositorError_None) {
				LOG_DEBUG << "OpenVR submit failed: " << error;
			}
			g_openVr.PostCompositorWorkCall(true);
			return error;
		}

		template<typename CallOriginal>
		vr::EVRCompositorError ProcessWaitGetPoses(CallOriginal callOriginal) {
			DepthGuard guard(t_waitGetPosesDepth);
			if (!guard.IsOutermost()) {
				return callOriginal();
			}

			g_openVr.PreWaitGetPoses();
			auto error = callOriginal();
			g_openVr.PostWaitGetPoses();
			if (error != vr::VRCompositorError_None) {
				LOG_DEBUG << "OpenVR WaitGetPoses failed: " << error;
			}
			return error;
		}

		template<typename CallOriginal>
		void ProcessPostPresentHandoff(CallOriginal callOriginal) {
			DepthGuard guard(t_postPresentHandoffDepth);
			if (!guard.IsOutermost()) {
				callOriginal();
				return;
			}

			g_openVr.PreCompositorWorkCall();
			callOriginal();
			g_openVr.PostCompositorWorkCall();
		}

		template<typename CallOriginal>
		void ProcessGetRecommendedRenderTargetSize(uint32_t *width, uint32_t *height, CallOriginal callOriginal) {
			DepthGuard guard(t_renderTargetSizeDepth);
			callOriginal(width, height);
			if (!guard.IsOutermost() || width == nullptr || height == nullptr) {
				return;
			}

			AdjustRenderResolution(*width, *height);
		}

		// ---------------------------------------------------------------------------------------
		// Hook slots. Each slot is a distinct detour with its own trampoline, so several different
		// target functions (one per interface version / FnTable) can be hooked at the same time.
		// ---------------------------------------------------------------------------------------
		constexpr int kSlotCount = 8;

		using VtableSubmitFn = vr::EVRCompositorError (*)(void *self, vr::EVREye, const vr::Texture_t *, const vr::VRTextureBounds_t *, vr::EVRSubmitFlags);
		using FnTableSubmitFn = vr::EVRCompositorError (*)(vr::EVREye, const vr::Texture_t *, const vr::VRTextureBounds_t *, vr::EVRSubmitFlags);
		using VtableWaitGetPosesFn = vr::EVRCompositorError (*)(void *self, vr::TrackedDevicePose_t *, uint32_t, vr::TrackedDevicePose_t *, uint32_t);
		using FnTableWaitGetPosesFn = vr::EVRCompositorError (*)(vr::TrackedDevicePose_t *, uint32_t, vr::TrackedDevicePose_t *, uint32_t);
		using VtablePostPresentHandoffFn = void (*)(void *self);
		using FnTablePostPresentHandoffFn = void (*)();
		using VtableRenderTargetSizeFn = void (*)(void *self, uint32_t *, uint32_t *);
		using FnTableRenderTargetSizeFn = void (*)(uint32_t *, uint32_t *);

		template<int Slot>
		struct VtableSubmitHook {
			static inline VtableSubmitFn original = nullptr;
			static inline std::atomic<bool> logged { false };
			static vr::EVRCompositorError Detour(void *self, vr::EVREye eye, const vr::Texture_t *texture, const vr::VRTextureBounds_t *bounds, vr::EVRSubmitFlags submitFlags) {
				LogFirstCall(logged, "IVRCompositor::Submit (vtable)", Slot);
				return ProcessSubmit(eye, texture, bounds, submitFlags, [self](vr::EVREye e, const vr::Texture_t *t, const vr::VRTextureBounds_t *b, vr::EVRSubmitFlags f) {
					return original(self, e, t, b, f);
				});
			}
		};

		template<int Slot>
		struct FnTableSubmitHook {
			static inline FnTableSubmitFn original = nullptr;
			static inline std::atomic<bool> logged { false };
			static vr::EVRCompositorError Detour(vr::EVREye eye, const vr::Texture_t *texture, const vr::VRTextureBounds_t *bounds, vr::EVRSubmitFlags submitFlags) {
				LogFirstCall(logged, "IVRCompositor::Submit (FnTable)", Slot);
				return ProcessSubmit(eye, texture, bounds, submitFlags, [](vr::EVREye e, const vr::Texture_t *t, const vr::VRTextureBounds_t *b, vr::EVRSubmitFlags f) {
					return original(e, t, b, f);
				});
			}
		};

		template<int Slot>
		struct VtableWaitGetPosesHook {
			static inline VtableWaitGetPosesFn original = nullptr;
			static inline std::atomic<bool> logged { false };
			static vr::EVRCompositorError Detour(void *self, vr::TrackedDevicePose_t *renderPoses, uint32_t renderPoseCount, vr::TrackedDevicePose_t *gamePoses, uint32_t gamePoseCount) {
				LogFirstCall(logged, "IVRCompositor::WaitGetPoses (vtable)", Slot);
				return ProcessWaitGetPoses([&]() { return original(self, renderPoses, renderPoseCount, gamePoses, gamePoseCount); });
			}
		};

		template<int Slot>
		struct FnTableWaitGetPosesHook {
			static inline FnTableWaitGetPosesFn original = nullptr;
			static inline std::atomic<bool> logged { false };
			static vr::EVRCompositorError Detour(vr::TrackedDevicePose_t *renderPoses, uint32_t renderPoseCount, vr::TrackedDevicePose_t *gamePoses, uint32_t gamePoseCount) {
				LogFirstCall(logged, "IVRCompositor::WaitGetPoses (FnTable)", Slot);
				return ProcessWaitGetPoses([&]() { return original(renderPoses, renderPoseCount, gamePoses, gamePoseCount); });
			}
		};

		template<int Slot>
		struct VtablePostPresentHandoffHook {
			static inline VtablePostPresentHandoffFn original = nullptr;
			static inline std::atomic<bool> logged { false };
			static void Detour(void *self) {
				LogFirstCall(logged, "IVRCompositor::PostPresentHandoff (vtable)", Slot);
				ProcessPostPresentHandoff([&]() { original(self); });
			}
		};

		template<int Slot>
		struct FnTablePostPresentHandoffHook {
			static inline FnTablePostPresentHandoffFn original = nullptr;
			static inline std::atomic<bool> logged { false };
			static void Detour() {
				LogFirstCall(logged, "IVRCompositor::PostPresentHandoff (FnTable)", Slot);
				ProcessPostPresentHandoff([]() { original(); });
			}
		};

		template<int Slot>
		struct VtableRenderTargetSizeHook {
			static inline VtableRenderTargetSizeFn original = nullptr;
			static inline std::atomic<bool> logged { false };
			static void Detour(void *self, uint32_t *width, uint32_t *height) {
				LogFirstCall(logged, "IVRSystem::GetRecommendedRenderTargetSize (vtable)", Slot);
				ProcessGetRecommendedRenderTargetSize(width, height, [self](uint32_t *w, uint32_t *h) { original(self, w, h); });
			}
		};

		template<int Slot>
		struct FnTableRenderTargetSizeHook {
			static inline FnTableRenderTargetSizeFn original = nullptr;
			static inline std::atomic<bool> logged { false };
			static void Detour(uint32_t *width, uint32_t *height) {
				LogFirstCall(logged, "IVRSystem::GetRecommendedRenderTargetSize (FnTable)", Slot);
				ProcessGetRecommendedRenderTargetSize(width, height, [](uint32_t *w, uint32_t *h) { original(w, h); });
			}
		};

		struct SlotEntry {
			void *detour;
			void **original;
		};

		template<template<int> class Hook, int... Slots>
		std::array<SlotEntry, sizeof...(Slots)> MakeSlotTable(std::integer_sequence<int, Slots...>) {
			return { { SlotEntry { reinterpret_cast<void *>(&Hook<Slots>::Detour), reinterpret_cast<void **>(&Hook<Slots>::original) }... } };
		}

		struct SlotPool {
			std::array<SlotEntry, kSlotCount> slots;
			int used = 0;
		};

		template<template<int> class Hook>
		SlotPool &GetSlotPool() {
			static SlotPool pool { MakeSlotTable<Hook>(std::make_integer_sequence<int, kSlotCount>{}), 0 };
			return pool;
		}

		// Hooks `target` with the next free detour from `pool`. Skips targets that are already hooked.
		void InstallSlotHook(const char *name, SlotPool &pool, void *target) {
			if (target == nullptr) {
				LOG_ERROR << "  " << name << ": target is null, not hooking";
				return;
			}
			if (std::find(g_slotHookTargets.begin(), g_slotHookTargets.end(), target) != g_slotHookTargets.end()) {
				LOG_INFO << "  " << name << " at " << target << " is already hooked";
				return;
			}
			if (pool.used >= kSlotCount) {
				LOG_ERROR << "  " << name << ": out of hook slots, not hooking " << target;
				return;
			}

			SlotEntry &slot = pool.slots[pool.used];
			MH_STATUS createResult = MH_CreateHook(target, slot.detour, slot.original);
			if (createResult != MH_OK) {
				LOG_ERROR << "  " << name << ": MH_CreateHook failed for " << target << " (" << createResult << ")";
				return;
			}
			MH_STATUS enableResult = MH_EnableHook(target);
			if (enableResult != MH_OK) {
				LOG_ERROR << "  " << name << ": MH_EnableHook failed for " << target << " (" << enableResult << ")";
				MH_RemoveHook(target);
				return;
			}

			LOG_INFO << "  Hooked " << name << " at " << target << " using slot " << pool.used;
			g_slotHookTargets.push_back(target);
			++pool.used;
		}

		void RemoveAllSlotHooks() {
			for (void *target : g_slotHookTargets) {
				MH_DisableHook(target);
				MH_RemoveHook(target);
			}
			g_slotHookTargets.clear();
		}

		// ---------------------------------------------------------------------------------------
		// Method positions per interface version (identical for the C++ vtable and the FnTable).
		// ---------------------------------------------------------------------------------------
		struct CompositorLayout {
			int waitGetPoses = -1;
			int submit = -1;
			int postPresentHandoff = -1;
		};

		bool GetCompositorLayout(unsigned int version, CompositorLayout &layout) {
			if (version >= 9 && version < 12) {
				layout = { -1, 4, -1 };
				return true;
			}
			if (version >= 12 && version < 15) {
				layout = { -1, 5, -1 };
				return true;
			}
			if (version >= 15 && version <= 27) {
				layout = { 2, 5, 7 };
				return true;
			}
			if (version == 28) {
				layout = { 2, 5, 8 };  // SubmitWithArrayIndex inserted at 6
				return true;
			}
			if (version == 29) {
				layout = { 2, 6, 9 };  // GetSubmitTexture inserted at 5
				return true;
			}
			return false;
		}

		void **GetVtable(void *instance) {
			return *reinterpret_cast<void ***>(instance);
		}

		// Legacy (pre-v9) compositor submit signatures, kept from upstream. Single-target only.
		vr::EVRCompositorError IVRCompositor008Hook_Submit(vr::IVRCompositor *self, vr::EVREye eEye, unsigned int eTextureType, void *pTexture, const vr::VRTextureBounds_t *pBounds, vr::EVRSubmitFlags nSubmitFlags) {
			vr::Texture_t texInfo { pTexture, (vr::ETextureType)eTextureType, vr::ColorSpace_Auto };
			OpenVrSubmitInfo info { eEye, &texInfo, pBounds, nSubmitFlags };
			g_openVr.OnSubmit(info);
			g_openVr.PreCompositorWorkCall(true);
			auto error = hooks::CallOriginal(IVRCompositor008Hook_Submit)(self, info.eye, info.texture->eType, info.texture->handle, info.bounds, info.submitFlags);
			g_openVr.PostCompositorWorkCall(true);
			return error;
		}

		vr::EVRCompositorError IVRCompositor007Hook_Submit(vr::IVRCompositor *self, vr::EVREye eEye, unsigned int eTextureType, void *pTexture, const vr::VRTextureBounds_t *pBounds) {
			vr::Texture_t texInfo { pTexture, (vr::ETextureType)eTextureType, vr::ColorSpace_Auto };
			OpenVrSubmitInfo info { eEye, &texInfo, pBounds, vr::Submit_Default };
			g_openVr.OnSubmit(info);
			g_openVr.PreCompositorWorkCall(true);
			auto error = hooks::CallOriginal(IVRCompositor007Hook_Submit)(self, info.eye, info.texture->eType, info.texture->handle, info.bounds);
			g_openVr.PostCompositorWorkCall(true);
			return error;
		}

		void HookCompositor(unsigned int version, void *instance, bool isFnTable) {
			if (!isFnTable && (version == 7 || version == 8)) {
				uint32_t methodPos = 6;
				hooks::InstallVirtualFunctionHook("IVRCompositor::Submit (legacy)", instance, methodPos,
					version == 8 ? (void*)&IVRCompositor008Hook_Submit : (void*)&IVRCompositor007Hook_Submit);
				return;
			}

			CompositorLayout layout;
			if (!GetCompositorLayout(version, layout)) {
				LOG_ERROR << "Don't know how to inject into version " << version << " of IVRCompositor" << (isFnTable ? " (FnTable)" : "");
				return;
			}

			void **methods = isFnTable ? reinterpret_cast<void **>(instance) : GetVtable(instance);
			if (isFnTable) {
				InstallSlotHook("IVRCompositor::Submit (FnTable)", GetSlotPool<FnTableSubmitHook>(), methods[layout.submit]);
				if (layout.waitGetPoses >= 0) {
					InstallSlotHook("IVRCompositor::WaitGetPoses (FnTable)", GetSlotPool<FnTableWaitGetPosesHook>(), methods[layout.waitGetPoses]);
				}
				if (layout.postPresentHandoff >= 0) {
					InstallSlotHook("IVRCompositor::PostPresentHandoff (FnTable)", GetSlotPool<FnTablePostPresentHandoffHook>(), methods[layout.postPresentHandoff]);
				}
			}
			else {
				InstallSlotHook("IVRCompositor::Submit (vtable)", GetSlotPool<VtableSubmitHook>(), methods[layout.submit]);
				if (layout.waitGetPoses >= 0) {
					InstallSlotHook("IVRCompositor::WaitGetPoses (vtable)", GetSlotPool<VtableWaitGetPosesHook>(), methods[layout.waitGetPoses]);
				}
				if (layout.postPresentHandoff >= 0) {
					InstallSlotHook("IVRCompositor::PostPresentHandoff (vtable)", GetSlotPool<VtablePostPresentHandoffHook>(), methods[layout.postPresentHandoff]);
				}
			}
		}

		void HookSystem(unsigned int version, void *instance, bool isFnTable) {
			int methodPos = version >= 9 ? 0 : 1;
			if (isFnTable) {
				void **methods = reinterpret_cast<void **>(instance);
				InstallSlotHook("IVRSystem::GetRecommendedRenderTargetSize (FnTable)", GetSlotPool<FnTableRenderTargetSizeHook>(), methods[methodPos]);
			}
			else {
				InstallSlotHook("IVRSystem::GetRecommendedRenderTargetSize (vtable)", GetSlotPool<VtableRenderTargetSizeHook>(), GetVtable(instance)[methodPos]);
			}
		}

		void *Hook_VRClientCoreFactory(const char *pInterfaceName, int *pReturnCode) {
			void *instance = hooks::CallOriginal(Hook_VRClientCoreFactory)(pInterfaceName, pReturnCode);
			HookOpenVrInterface(pInterfaceName, instance);
			return instance;
		}

		void *IVRClientCoreHook_GetGenericInterface(void *self, const char *interfaceName, vr::EVRInitError *error) {
			void *instance = hooks::CallOriginal(IVRClientCoreHook_GetGenericInterface)(self, interfaceName, error);
			HookOpenVrInterface(interfaceName, instance);
			return instance;
		}

		void IVRClientCoreHook_Cleanup(void *self) {
			hooks::CallOriginal(IVRClientCoreHook_Cleanup)(self);
			LOG_INFO << "IVRClientCore::Cleanup was called, deleting hooks...";
			std::lock_guard<std::recursive_mutex> lock(g_interfaceHookMutex);
			hooks::RemoveHook((void*)&IVRClientCoreHook_GetGenericInterface);
			hooks::RemoveHook((void*)&IVRClientCoreHook_Cleanup);
			hooks::RemoveHook((void*)IVRCompositor008Hook_Submit);
			hooks::RemoveHook((void*)IVRCompositor007Hook_Submit);
			RemoveAllSlotHooks();
			// Slot pools are not reset: their trampolines were freed by MH_RemoveHook and the slots
			// are simply not reused, which keeps any in-flight calls on other threads safe.
		}
	}

	void InstallOpenVrHooks() {
		static bool hooksLoaded = false;
		if (hooksLoaded) {
			return;
		}

#ifdef WIN64
		std::wstring dllName = L"vrclient_x64.dll";
#else
		std::wstring dllName = L"vrclient.dll";
#endif
		HMODULE handle;
		if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, dllName.c_str(), &handle) || handle == g_moduleSelf) {
			return;
		}

		LOG_INFO << dllName << " is loaded in the process, installing hooks...";
		hooks::InstallHookInDll("VRClientCoreFactory", handle, (void*)Hook_VRClientCoreFactory);

		hooksLoaded = true;
	}

	void HookOpenVrInterface(const char *interfaceName, void *instance) {
		LOG_INFO << "OpenVR: requested interface " << interfaceName;
		if (instance == nullptr || interfaceName == nullptr) {
			return;
		}

		std::lock_guard<std::recursive_mutex> lock(g_interfaceHookMutex);

		if (unsigned int version = 0; std::sscanf(interfaceName, "IVRClientCore_%u", &version) == 1) {
			hooks::RemoveHook((void*)&IVRClientCoreHook_Cleanup);
			hooks::RemoveHook((void*)&IVRClientCoreHook_GetGenericInterface);
			if (version <= 3) {
				hooks::InstallVirtualFunctionHook("IVRClientCore::GetGenericInterface", instance, 3, (void*)&IVRClientCoreHook_GetGenericInterface);
				hooks::InstallVirtualFunctionHook("IVRClientCore::Cleanup", instance, 1, (void*)&IVRClientCoreHook_Cleanup);
				g_clientCoreInstance = instance;
			}
			else {
				LOG_ERROR << "Don't know how to inject into version " << version << " of IVRClientCore";
			}
			return;
		}

		const char fnTablePrefix[] = "FnTable:";
		bool isFnTable = std::strncmp(interfaceName, fnTablePrefix, sizeof(fnTablePrefix) - 1) == 0;
		const char *baseName = isFnTable ? interfaceName + sizeof(fnTablePrefix) - 1 : interfaceName;

		if (unsigned int version = 0; std::sscanf(baseName, "IVRCompositor_%u", &version) == 1) {
			HookCompositor(version, instance, isFnTable);
		}
		else if (unsigned int version = 0; std::sscanf(baseName, "IVRSystem_%u", &version) == 1) {
			HookSystem(version, instance, isFnTable);
		}
	}

	vr::IVRCompositor * GetOpenVrCompositor() {
		if (g_clientCoreInstance == nullptr) {
			return nullptr;
		}

		return (vr::IVRCompositor*)IVRClientCoreHook_GetGenericInterface(g_clientCoreInstance, vr::IVRCompositor_Version, nullptr);
	}

	vr::IVRSystem * GetOpenVrSystem() {
		if (g_clientCoreInstance == nullptr) {
			return nullptr;
		}

		return (vr::IVRSystem*)IVRClientCoreHook_GetGenericInterface(g_clientCoreInstance, vr::IVRSystem_Version, nullptr);
	}
}

// ResoniteFSR3: called from the renderer plugin (P/Invoke) with Unity's current XRSettings.renderViewportScale.
// 1.0 disables bounds expansion.
extern "C" __declspec(dllexport) void ResoniteFSR3_SetSubmitViewportScale(float scale) {
	vrperfkit::g_submitViewportScale.store(scale);
}

// ResoniteFSR3: register (or clear with null) the full-resolution FSR 3.1 output texture (ID3D11Texture2D*) for an eye.
extern "C" __declspec(dllexport) void ResoniteFSR3_SetSubmitTextureOverride(int eyeIndex, void *texture) {
	if (eyeIndex == 0 || eyeIndex == 1) {
		vrperfkit::g_submitTextureOverride[eyeIndex].store(texture);
	}
}

// ResoniteFSR3: present the override textures vertically flipped (non-zero) or as stored (zero).
extern "C" __declspec(dllexport) void ResoniteFSR3_SetSubmitFlipVertical(int flip) {
	vrperfkit::g_submitOverrideFlipVertical.store(flip);
}
