using System.IO;
using Unity.Pipeline.Commands;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DepthTerrain.Editor
{
    /// <summary>
    /// Custom Unity Pipeline CLI commands (`unity command ...`) for automated scene setup,
    /// Depth Pro 2 dataset loading, camera framing, and verification.
    /// </summary>
    public static class DepthTerrainCliCommands
    {
        [CliCommand("depth_terrain_setup_scene", "Create and save the URP DepthTerrainShowcase scene", MainThreadRequired = true)]
        public static string SetupShowcaseScene(
            [CliArg("dataset", "Bundled sample name or folder path to load")] string dataset = "grand_canyon_real_satellite"
        )
        {
            var ctrl = DepthTerrainSceneSetup.CreateOrUpdateShowcaseScene(dataset, saveScene: true);
            return $"Showcase scene saved at {DepthTerrainSceneSetup.ShowcaseScenePath}. Active dataset: {ctrl.activeDatasetFolder}. Status: {ctrl.StatusMessage}";
        }

        [CliCommand("depth_terrain_load", "Load a Depth Pro 2 dataset onto the active scene's DepthTerrainController", MainThreadRequired = true)]
        public static string LoadDatasetCommand(
            [CliArg("dataset", "Preset name in StreamingAssets/DepthTerrainSamples or absolute path")] string dataset,
            [CliArg("height", "Optional terrain elevation scale in meters (<= 0 uses preset recommendation)")] float height = -1f,
            [CliArg("water", "Optional water level [0..0.75] (< 0 uses preset recommendation)")] float water = -1f,
            [CliArg("save", "Save active scene after loading")] bool save = false
        )
        {
            var ctrl = Object.FindFirstObjectByType<DepthTerrainController>();
            if (ctrl == null)
            {
                ctrl = DepthTerrainSceneSetup.CreateOrUpdateShowcaseScene(dataset, saveScene: save);
            }
            else
            {
                ctrl.LoadDataset(dataset, applyRecommendedSettings: true);
            }

            bool dirty = false;
            if (height > 0f)
            {
                ctrl.buildSettings.terrainHeight = height;
                dirty = true;
            }
            if (water >= 0f)
            {
                ctrl.buildSettings.waterLevel = water;
                ctrl.buildSettings.showWaterPlane = water > 0.005f;
                dirty = true;
            }
            if (dirty)
            {
                ctrl.RebuildTerrainFromCache();
            }

            if (ctrl.orbitCamera != null)
            {
                ctrl.orbitCamera.FrameTerrain(ctrl.buildSettings.terrainSize, ctrl.buildSettings.terrainHeight, immediate: true);
            }

            if (save)
            {
                EditorSceneManager.SaveOpenScenes();
                AssetDatabase.SaveAssets();
            }

            return $"Loaded '{dataset}' -> {ctrl.StatusMessage}";
        }

        [CliCommand("depth_terrain_frame_camera", "Position the Main Camera at a scenic angle around the terrain", MainThreadRequired = true)]
        public static string FrameCameraCommand(
            [CliArg("yaw", "Orbit yaw angle in degrees")] float yaw = 38f,
            [CliArg("pitch", "Orbit pitch angle in degrees")] float pitch = 34f,
            [CliArg("distance", "Distance from terrain center (<= 0 auto-fits)")] float distance = -1f
        )
        {
            var ctrl = Object.FindFirstObjectByType<DepthTerrainController>();
            var cam = Object.FindFirstObjectByType<OrbitFlyCamera>();
            if (cam == null) return "No OrbitFlyCamera found in scene.";

            float size = ctrl != null ? ctrl.buildSettings.terrainSize : 500f;
            float height = ctrl != null ? ctrl.buildSettings.terrainHeight : 120f;
            cam.FrameTerrain(size, height, yaw, pitch, immediate: true);
            if (distance > 0f)
            {
                cam.distance = distance;
                cam.SyncTargetsFromFields();
                cam.ApplyTransformImmediate();
            }
            return $"Camera framed at yaw={yaw}, pitch={pitch}, dist={cam.distance:F1}.";
        }

        [CliCommand("depth_terrain_capture_game_png", "Render Main Camera offscreen directly to a crisp PNG file for visual verification", MainThreadRequired = true)]
        public static string CaptureGameCameraPng(
            [CliArg("output", "Output PNG file path")] string output,
            [CliArg("width", "Image width in pixels")] int width = 1280,
            [CliArg("height", "Image height in pixels")] int height = 720
        )
        {
            var cam = Camera.main ?? Object.FindFirstObjectByType<Camera>();
            if (cam == null) return "No Camera found in scene.";

            var rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
            var prevTarget = cam.targetTexture;
            var prevActive = RenderTexture.active;

            try
            {
                cam.targetTexture = rt;
                cam.Render();
                RenderTexture.active = rt;
                var tex = new Texture2D(width, height, TextureFormat.RGB24, false);
                tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
                tex.Apply();

                string dir = Path.GetDirectoryName(output);
                if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
                File.WriteAllBytes(output, tex.EncodeToPNG());
                Object.DestroyImmediate(tex);
                return $"Captured {width}x{height} render to {output}";
            }
            finally
            {
                cam.targetTexture = prevTarget;
                RenderTexture.active = prevActive;
                rt.Release();
                Object.DestroyImmediate(rt);
            }
        }
    }
}
