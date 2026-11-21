using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.SceneManagement;

namespace DepthTerrain.Editor
{
    /// <summary>
    /// Creates and configures the URP DepthTerrainShowcase scene cleanly inside the Unity Editor.
    /// </summary>
    public static class DepthTerrainSceneSetup
    {
        public const string ShowcaseScenePath = "Assets/Scenes/DepthTerrainShowcase.unity";
        public const string VolumeProfilePath = "Assets/DepthTerrain/DepthTerrainVolumeProfile.asset";

        [MenuItem("Tools/Depth Terrain/Create Showcase Scene", priority = 20)]
        public static DepthTerrainController CreateOrUpdateShowcaseSceneMenu()
        {
            return CreateOrUpdateShowcaseScene(defaultDataset: "grand_canyon_real_satellite", saveScene: true);
        }

        public static DepthTerrainController CreateOrUpdateShowcaseScene(string defaultDataset = "grand_canyon_real_satellite", bool saveScene = true)
        {
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // Atmospheric fog for depth perception across mountain ranges & canyons
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogColor = new Color(0.66f, 0.76f, 0.86f, 1f);
            RenderSettings.fogDensity = 0.00055f;
            RenderSettings.ambientMode = AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.55f, 0.68f, 0.85f, 1f);
            RenderSettings.ambientEquatorColor = new Color(0.45f, 0.48f, 0.50f, 1f);
            RenderSettings.ambientGroundColor = new Color(0.24f, 0.22f, 0.20f, 1f);

            // Golden-hour Directional Sun Light to sculpt 3D terrain shadows dramatically
            var sunGo = new GameObject("Directional Sun");
            var sunLight = sunGo.AddComponent<Light>();
            sunLight.type = LightType.Directional;
            sunLight.color = new Color(1.0f, 0.94f, 0.84f, 1f);
            sunLight.intensity = 2.2f;
            sunLight.shadows = LightShadows.Soft;
            sunLight.shadowStrength = 0.82f;
            sunGo.transform.rotation = Quaternion.Euler(34f, -42f, 0f);
            RenderSettings.sun = sunLight;

            // Main Camera with OrbitFlyCamera + URP camera data
            var camGo = new GameObject("Main Camera") { tag = "MainCamera" };
            var cam = camGo.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.Skybox;
            cam.backgroundColor = new Color(0.52f, 0.66f, 0.82f, 1f);
            cam.fieldOfView = 45f;
            cam.nearClipPlane = 0.5f;
            cam.farClipPlane = 6000f;
            var urpCamData = camGo.AddComponent<UniversalAdditionalCameraData>();
            urpCamData.renderPostProcessing = true;
            urpCamData.antialiasing = AntialiasingMode.SubpixelMorphologicalAntiAliasing;
            camGo.AddComponent<AudioListener>();
            var orbitCam = camGo.AddComponent<OrbitFlyCamera>();
            orbitCam.FrameTerrain(500f, 125f, yaw: 38f, pitch: 34f, immediate: true);

            // Global URP Post-Processing Volume (ACES Tonemapping + subtle Bloom + Vignette)
            var volumeGo = new GameObject("Global PostProcess Volume");
            var volume = volumeGo.AddComponent<Volume>();
            volume.isGlobal = true;
            volume.sharedProfile = GetOrCreateVolumeProfile();

            // DepthTerrain Root Controller
            var rootGo = new GameObject("DepthTerrainSystem");
            var controller = rootGo.AddComponent<DepthTerrainController>();
            controller.orbitCamera = orbitCam;
            controller.sunLight = sunLight;
            controller.activeDatasetFolder = defaultDataset;

            // Attach RuntimeTerrainHUD if present in Runtime assembly
            var hudType = typeof(DepthTerrainController).Assembly.GetType("DepthTerrain.RuntimeTerrainHUD");
            if (hudType != null)
            {
                rootGo.AddComponent(hudType);
            }

            // Load default dataset immediately if available on disk
            if (!string.IsNullOrWhiteSpace(defaultDataset))
            {
                string resolved = DepthTerrainController.ResolveDatasetPath(defaultDataset);
                if (File.Exists(Path.Combine(resolved, "metadata.json")))
                {
                    controller.LoadDataset(defaultDataset, applyRecommendedSettings: true);
                }
            }

            if (saveScene)
            {
                EnsureDirectory("Assets/Scenes");
                EditorSceneManager.SaveScene(scene, ShowcaseScenePath);
                RegisterSceneInBuildSettings(ShowcaseScenePath);
                AssetDatabase.SaveAssets();
            }

            return controller;
        }

        private static VolumeProfile GetOrCreateVolumeProfile()
        {
            var existing = AssetDatabase.LoadAssetAtPath<VolumeProfile>(VolumeProfilePath);
            if (existing != null)
                return existing;

            EnsureDirectory(Path.GetDirectoryName(VolumeProfilePath));
            var profile = ScriptableObject.CreateInstance<VolumeProfile>();

            var tonemapping = profile.Add<Tonemapping>(overrides: true);
            tonemapping.mode.overrideState = true;
            tonemapping.mode.value = TonemappingMode.ACES;

            var colorAdjust = profile.Add<ColorAdjustments>(overrides: true);
            colorAdjust.contrast.overrideState = true;
            colorAdjust.contrast.value = 8f;
            colorAdjust.saturation.overrideState = true;
            colorAdjust.saturation.value = 10f;

            var bloom = profile.Add<Bloom>(overrides: true);
            bloom.threshold.overrideState = true;
            bloom.threshold.value = 1.05f;
            bloom.intensity.overrideState = true;
            bloom.intensity.value = 0.25f;

            var vignette = profile.Add<Vignette>(overrides: true);
            vignette.intensity.overrideState = true;
            vignette.intensity.value = 0.16f;

            AssetDatabase.CreateAsset(profile, VolumeProfilePath);
            AssetDatabase.SaveAssets();
            return profile;
        }

        private static void EnsureDirectory(string assetFolder)
        {
            if (string.IsNullOrEmpty(assetFolder)) return;
            string full = Path.GetFullPath(assetFolder);
            if (!Directory.Exists(full))
            {
                Directory.CreateDirectory(full);
                AssetDatabase.Refresh();
            }
        }

        private static void RegisterSceneInBuildSettings(string scenePath)
        {
            var existing = EditorBuildSettings.scenes;
            foreach (var s in existing)
            {
                if (s.path == scenePath)
                    return;
            }
            var updated = new EditorBuildSettingsScene[existing.Length + 1];
            updated[0] = new EditorBuildSettingsScene(scenePath, true);
            for (int i = 0; i < existing.Length; i++)
                updated[i + 1] = existing[i];
            EditorBuildSettings.scenes = updated;
        }
    }
}
