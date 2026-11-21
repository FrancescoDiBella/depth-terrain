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
        public const string SkyboxMaterialPath = "Assets/DepthTerrain/DepthTerrainSkybox.mat";

        [MenuItem("Tools/Depth Terrain/Create Showcase Scene", priority = 20)]
        public static DepthTerrainController CreateOrUpdateShowcaseSceneMenu()
        {
            return CreateOrUpdateShowcaseScene(defaultDataset: "grand_canyon_real_satellite", saveScene: true);
        }

        public static DepthTerrainController CreateOrUpdateShowcaseScene(string defaultDataset = "grand_canyon_real_satellite", bool saveScene = true)
        {
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // Procedural Skybox + crisp daylight atmospheric fog
            RenderSettings.skybox = GetOrCreateSkyboxMaterial();
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogColor = new Color(0.74f, 0.84f, 0.94f, 1f);
            RenderSettings.fogDensity = 0.00034f;
            RenderSettings.ambientMode = AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.60f, 0.74f, 0.90f, 1f);
            RenderSettings.ambientEquatorColor = new Color(0.52f, 0.56f, 0.60f, 1f);
            RenderSettings.ambientGroundColor = new Color(0.28f, 0.26f, 0.24f, 1f);

            // Golden-hour Directional Sun Light to sculpt 3D terrain shadows dramatically
            var sunGo = new GameObject("Directional Sun");
            var sunLight = sunGo.AddComponent<Light>();
            sunLight.type = LightType.Directional;
            sunLight.color = new Color(1.0f, 0.95f, 0.87f, 1f);
            sunLight.intensity = 2.0f;
            sunLight.shadows = LightShadows.Soft;
            sunLight.shadowStrength = 0.78f;
            sunGo.transform.rotation = Quaternion.Euler(36f, -42f, 0f);
            RenderSettings.sun = sunLight;
            DynamicGI.UpdateEnvironment();

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

        private static Material GetOrCreateSkyboxMaterial()
        {
            var existing = AssetDatabase.LoadAssetAtPath<Material>(SkyboxMaterialPath);
            if (existing != null)
                return existing;

            EnsureDirectory(Path.GetDirectoryName(SkyboxMaterialPath));
            var shader = Shader.Find("Skybox/Procedural") ?? Shader.Find("Skybox/6 Sided");
            var mat = new Material(shader) { name = "DepthTerrainSkybox" };
            if (shader != null && shader.name == "Skybox/Procedural")
            {
                mat.SetFloat("_SunDisk", 2f);
                mat.SetFloat("_SunSize", 0.04f);
                mat.SetFloat("_SunSizeConvergence", 5f);
                mat.SetFloat("_AtmosphereThickness", 0.85f);
                mat.SetColor("_SkyTint", new Color(0.42f, 0.58f, 0.78f, 1f));
                mat.SetColor("_GroundColor", new Color(0.32f, 0.35f, 0.40f, 1f));
                mat.SetFloat("_Exposure", 1.15f);
            }
            AssetDatabase.CreateAsset(mat, SkyboxMaterialPath);
            AssetDatabase.SaveAssets();
            return mat;
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
