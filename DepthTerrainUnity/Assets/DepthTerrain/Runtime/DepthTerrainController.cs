using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace DepthTerrain
{
    /// <summary>
    /// Scene-level orchestrator that loads Apple Depth Pro 2 terrain bundles (from StreamingAssets presets
    /// or live pipeline runs), applies real-time non-destructive terrain shaping, and supports aerial video frame playback.
    /// </summary>
    [ExecuteAlways]
    public class DepthTerrainController : MonoBehaviour
    {
        [Header("Scene References")]
        public Terrain targetTerrain;
        public GameObject waterPlane;
        public OrbitFlyCamera orbitCamera;
        public Light sunLight;

        [Header("Dataset & Terrain Settings")]
        [Tooltip("Relative path under StreamingAssets/DepthTerrainSamples or absolute path to an exported Depth Pro 2 folder.")]
        public string activeDatasetFolder = "vesuvius_real_satellite";
        public bool autoLoadOnStart = true;
        public TerrainBuildSettings buildSettings = new TerrainBuildSettings();

        [Header("Pipeline Bridge Overrides (Optional)")]
        public string customPipelineDirectory = "";
        public string customUvExecutable = "";

        // Cached loaded state for instant non-destructive rebuilds
        public DepthTerrainMetadata CurrentMetadata { get; private set; }
        public Texture2D CurrentAlbedoTexture { get; private set; }
        public Texture2D CurrentDepthPreviewTexture { get; private set; }
        public bool IsGenerating { get; private set; }
        public string StatusMessage { get; private set; } = "Ready";
        public int ActiveVideoFrameIndex { get; private set; }
        public int AvailableVideoFrames => _videoFrameFolders.Count;

        private float[,] _rawHeights;
        private readonly List<string> _videoFrameFolders = new List<string>();
        private CancellationTokenSource _activeCts;

        public event Action<DepthTerrainMetadata> OnTerrainRebuilt;
        public event Action<string> OnStatusChanged;

        private void Start()
        {
            if (Application.isPlaying && autoLoadOnStart && !string.IsNullOrWhiteSpace(activeDatasetFolder))
            {
                if (_rawHeights == null)
                {
                    LoadDataset(activeDatasetFolder, applyRecommendedSettings: true);
                }
            }
        }

        private void OnDisable()
        {
            CancelActiveGeneration();
        }

        /// <summary>
        /// Enumerate all bundled sample presets inside StreamingAssets/DepthTerrainSamples.
        /// </summary>
        public static string[] DiscoverBundledSamples()
        {
            string root = Path.Combine(Application.streamingAssetsPath, "DepthTerrainSamples");
            if (!Directory.Exists(root))
                return Array.Empty<string>();

            var valid = new List<string>();
            foreach (string dir in Directory.GetDirectories(root))
            {
                if (File.Exists(Path.Combine(dir, "metadata.json")))
                    valid.Add(Path.GetFileName(dir));
            }
            valid.Sort(StringComparer.OrdinalIgnoreCase);
            return valid.ToArray();
        }

        /// <summary>
        /// Resolve either an absolute dataset folder path or a preset name under StreamingAssets/DepthTerrainSamples.
        /// </summary>
        public static string ResolveDatasetPath(string folderOrPresetName)
        {
            if (string.IsNullOrWhiteSpace(folderOrPresetName))
                return "";

            if (Directory.Exists(folderOrPresetName) && File.Exists(Path.Combine(folderOrPresetName, "metadata.json")))
                return Path.GetFullPath(folderOrPresetName);

            string samplePath = Path.Combine(Application.streamingAssetsPath, "DepthTerrainSamples", folderOrPresetName);
            if (Directory.Exists(samplePath) && File.Exists(Path.Combine(samplePath, "metadata.json")))
                return samplePath;

            return folderOrPresetName;
        }

        /// <summary>
        /// Load a Depth Pro 2 output bundle folder and build the Unity Terrain immediately.
        /// </summary>
        public bool LoadDataset(string folderOrPresetName, bool applyRecommendedSettings = true)
        {
            string resolved = ResolveDatasetPath(folderOrPresetName);
            string metaPath = Path.Combine(resolved, "metadata.json");
            if (!File.Exists(metaPath))
            {
                SetStatus($"Dataset not found: {resolved}");
                Debug.LogWarning($"[DepthTerrain] metadata.json missing at '{metaPath}'.");
                return false;
            }

            try
            {
                var metadata = JsonUtility.FromJson<DepthTerrainMetadata>(File.ReadAllText(metaPath));
                string rawPath = Path.Combine(resolved, metadata.heightmapRawFile);
                string albedoPath = Path.Combine(resolved, metadata.albedoFile);
                string previewPath = Path.Combine(resolved, metadata.depthPreviewFile);

                if (!File.Exists(rawPath))
                {
                    SetStatus($"Heightmap RAW missing: {rawPath}");
                    return false;
                }

                byte[] rawBytes = File.ReadAllBytes(rawPath);
                bool littleEndian = !string.Equals(metadata.rawByteOrder, "big", StringComparison.OrdinalIgnoreCase);
                _rawHeights = HeightmapProcessor.DecodeRaw16(rawBytes, metadata.rawResolution, littleEndian, metadata.rawFlipY);

                CurrentAlbedoTexture = LoadTextureFromFile(albedoPath, "DepthTerrain_Albedo", sRGB: true);
                CurrentDepthPreviewTexture = File.Exists(previewPath)
                    ? LoadTextureFromFile(previewPath, "DepthTerrain_DepthPreview", sRGB: true)
                    : null;
                CurrentMetadata = metadata;
                activeDatasetFolder = folderOrPresetName;

                // Discover optional multi-frame aerial video sub-bundles
                _videoFrameFolders.Clear();
                ActiveVideoFrameIndex = 0;
                string framesRoot = Path.Combine(resolved, "frames");
                if (Directory.Exists(framesRoot))
                {
                    string[] subDirs = Directory.GetDirectories(framesRoot);
                    Array.Sort(subDirs, StringComparer.OrdinalIgnoreCase);
                    foreach (string sub in subDirs)
                    {
                        if (File.Exists(Path.Combine(sub, "metadata.json")))
                            _videoFrameFolders.Add(sub);
                    }
                }

                if (applyRecommendedSettings)
                {
                    if (metadata.recommendedTerrainSize > 10f)
                        buildSettings.terrainSize = metadata.recommendedTerrainSize;
                    if (metadata.recommendedTerrainHeight > 2f)
                        buildSettings.terrainHeight = metadata.recommendedTerrainHeight;
                    buildSettings.targetResolution = HeightmapProcessor.NearestUnityResolution(metadata.rawResolution);
                    buildSettings.waterLevel = Mathf.Clamp(metadata.recommendedWaterLevel, 0f, 0.75f);
                    buildSettings.showWaterPlane = buildSettings.waterLevel > 0.01f;
                }

                RebuildTerrainFromCache();

                if (orbitCamera != null)
                {
                    orbitCamera.FrameTerrain(buildSettings.terrainSize, buildSettings.terrainHeight);
                }

                SetStatus($"Loaded '{metadata.sourceLabel}' ({metadata.modelName}, {metadata.inferenceMs:F0} ms on {metadata.device})");
                return true;
            }
            catch (Exception ex)
            {
                SetStatus($"Load failed: {ex.Message}");
                Debug.LogException(ex);
                return false;
            }
        }

        /// <summary>
        /// Load a specific frame index from an aerial video dataset exported by `depth-terrain from-video`.
        /// </summary>
        public bool LoadVideoFrame(int frameIndex)
        {
            if (_videoFrameFolders.Count == 0)
                return false;

            int clamped = Mathf.Clamp(frameIndex, 0, _videoFrameFolders.Count - 1);
            string frameDir = _videoFrameFolders[clamped];
            string metaPath = Path.Combine(frameDir, "metadata.json");
            if (!File.Exists(metaPath)) return false;

            var metadata = JsonUtility.FromJson<DepthTerrainMetadata>(File.ReadAllText(metaPath));
            string rawPath = Path.Combine(frameDir, metadata.heightmapRawFile);
            string albedoPath = Path.Combine(frameDir, metadata.albedoFile);
            string previewPath = Path.Combine(frameDir, metadata.depthPreviewFile);

            byte[] rawBytes = File.ReadAllBytes(rawPath);
            bool littleEndian = !string.Equals(metadata.rawByteOrder, "big", StringComparison.OrdinalIgnoreCase);
            _rawHeights = HeightmapProcessor.DecodeRaw16(rawBytes, metadata.rawResolution, littleEndian, metadata.rawFlipY);
            CurrentAlbedoTexture = LoadTextureFromFile(albedoPath, "DepthTerrain_VideoAlbedo", sRGB: true);
            if (File.Exists(previewPath))
                CurrentDepthPreviewTexture = LoadTextureFromFile(previewPath, "DepthTerrain_VideoPreview", sRGB: true);
            CurrentMetadata = metadata;
            ActiveVideoFrameIndex = clamped;

            RebuildTerrainFromCache();
            SetStatus($"Video frame {clamped + 1}/{_videoFrameFolders.Count}: {metadata.sourceLabel}");
            return true;
        }

        /// <summary>
        /// Re-apply TerrainBuildSettings instantaneously without re-running Python inference.
        /// </summary>
        public void RebuildTerrainFromCache()
        {
            if (_rawHeights == null)
            {
                if (!string.IsNullOrWhiteSpace(activeDatasetFolder))
                    LoadDataset(activeDatasetFolder, applyRecommendedSettings: false);
                return;
            }

            float[,] processed = HeightmapProcessor.ProcessHeights(_rawHeights, buildSettings);
            var output = DepthTerrainBuilder.BuildOrUpdateTerrain(
                targetTerrain,
                transform,
                processed,
                CurrentAlbedoTexture,
                buildSettings,
                waterPlane
            );

            targetTerrain = output.Terrain;
            waterPlane = output.WaterPlane;
            OnTerrainRebuilt?.Invoke(CurrentMetadata);
        }

        /// <summary>
        /// Invoke Apple Depth Pro 2 asynchronously via local Python/uv pipeline, then immediately rebuild the Terrain.
        /// </summary>
        public async Task<PipelineRunResult> GenerateWithDepthPro2Async(PipelineRunRequest request, Action<string> onLogLine = null)
        {
            CancelActiveGeneration();
            _activeCts = new CancellationTokenSource();
            IsGenerating = true;
            SetStatus($"Running Apple Depth Pro 2 ({request.model.ToCliFlag()})...");

            try
            {
                var result = await DepthPipelineBridge.RunAsync(
                    request,
                    line =>
                    {
                        onLogLine?.Invoke(line);
                    },
                    customPipelineDirectory,
                    customUvExecutable,
                    _activeCts.Token
                );

                if (result.Success)
                {
                    LoadDataset(result.OutputFolder, applyRecommendedSettings: true);
                    SetStatus($"Generated '{result.Metadata.sourceLabel}' in {result.ElapsedSeconds:F1}s");
                }
                else
                {
                    SetStatus($"Pipeline error: {result.ErrorMessage}");
                    Debug.LogError($"[DepthTerrain] Pipeline failed:\n{result.ErrorMessage}\n{result.LogOutput}");
                }

                return result;
            }
            finally
            {
                IsGenerating = false;
            }
        }

        public void CancelActiveGeneration()
        {
            if (_activeCts != null)
            {
                _activeCts.Cancel();
                _activeCts.Dispose();
                _activeCts = null;
            }
        }

        private void SetStatus(string message)
        {
            StatusMessage = message;
            OnStatusChanged?.Invoke(message);
        }

        private static Texture2D LoadTextureFromFile(string path, string texName, bool sRGB)
        {
            if (!File.Exists(path)) return null;
            byte[] pngBytes = File.ReadAllBytes(path);
            var tex = new Texture2D(2, 2, TextureFormat.RGBA32, mipChain: true, linear: !sRGB)
            {
                name = texName,
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Trilinear,
                anisoLevel = 8
            };
            ImageConversion.LoadImage(tex, pngBytes, markNonReadable: false);
            return tex;
        }
    }
}
