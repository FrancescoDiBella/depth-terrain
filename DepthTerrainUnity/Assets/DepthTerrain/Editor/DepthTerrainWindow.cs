using System;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DepthTerrain.Editor
{
    /// <summary>
    /// Full-featured Unity Editor Studio for generating 3D Terrains from real-world maps,
    /// local top-down photos, procedural fictional worlds, and aerial videos using Apple Depth Pro 2.
    /// </summary>
    public class DepthTerrainWindow : EditorWindow
    {
        private static readonly string[] TabNames = { "Real Map", "Local Image", "Fictional World", "Aerial Video" };

        private static readonly (string Name, float Lat, float Lon, int Zoom, string Provider)[] RealPresets =
        {
            ("Grand Canyon, USA", 36.0998f, -112.1125f, 13, "esri_satellite"),
            ("Rome Colosseum, Italy", 41.8902f, 12.4922f, 17, "esri_satellite"),
            ("Mount Vesuvius, Italy", 40.8218f, 14.4260f, 14, "esri_satellite"),
            ("Dolomites (Tre Cime), Italy", 46.6187f, 12.3020f, 14, "esri_satellite"),
            ("Mount Fuji, Japan", 35.3606f, 138.7274f, 13, "esri_satellite"),
            ("Santorini Caldera, Greece", 36.4166f, 25.4324f, 13, "esri_satellite")
        };

        private DepthTerrainController _controller;
        private readonly PipelineRunRequest _request = new PipelineRunRequest();
        private Vector2 _scrollPos;
        private readonly StringBuilder _liveLogs = new StringBuilder();
        private int _selectedProviderIdx;
        private int _selectedBiomeIdx;
        private int _selectedSampleIdx;
        private string[] _bundledSamples = Array.Empty<string>();
        private WarmServerHealth _warmServerHealth = new WarmServerHealth();
        private bool _checkingServerHealth;

        [MenuItem("Tools/Depth Terrain/Depth Pro 2 Studio", priority = 10)]
        public static void ShowWindow()
        {
            var win = GetWindow<DepthTerrainWindow>("Depth Pro 2 Terrain Studio");
            win.minSize = new Vector2(460f, 640f);
            win.RefreshSamples();
            win.Show();
        }

        private void OnEnable()
        {
            RefreshSamples();
            FindOrBindController();
            RefreshWarmServerHealthAsync();
        }

        private void RefreshSamples()
        {
            _bundledSamples = DepthTerrainController.DiscoverBundledSamples();
            _selectedSampleIdx = Mathf.Clamp(_selectedSampleIdx, 0, Mathf.Max(0, _bundledSamples.Length - 1));
        }

        private void FindOrBindController()
        {
            if (_controller == null)
            {
                _controller = FindAnyObjectByType<DepthTerrainController>();
            }
        }

        private void OnGUI()
        {
            FindOrBindController();

            EditorGUILayout.Space(6);
            EditorGUILayout.LabelField("Apple Depth Pro 2 -> Unity 6 Terrain Studio", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Generate 3D Unity Terrains from real-world satellite/OSM maps, local top-down images, fictional worlds, or aerial videos using Apple Depth Pro 2.",
                MessageType.Info
            );

            _scrollPos = EditorGUILayout.BeginScrollView(_scrollPos);

            DrawSceneSection();
            EditorGUILayout.Space(8);
            DrawBundledSamplesSection();
            EditorGUILayout.Space(8);
            DrawSourceTabSection();
            EditorGUILayout.Space(8);
            DrawDepthPro2OptionsSection();
            EditorGUILayout.Space(8);
            DrawGenerateActions();
            EditorGUILayout.Space(8);
            DrawTerrainShapingSection();
            EditorGUILayout.Space(8);
            DrawPreviewAndLogsSection();

            EditorGUILayout.EndScrollView();
        }

        private void DrawSceneSection()
        {
            EditorGUILayout.LabelField("1. Scene Target", EditorStyles.boldLabel);
            using (new EditorGUILayout.HorizontalScope())
            {
                _controller = (DepthTerrainController)EditorGUILayout.ObjectField(
                    "Controller",
                    _controller,
                    typeof(DepthTerrainController),
                    allowSceneObjects: true
                );

                if (GUILayout.Button("Setup Showcase Scene", GUILayout.Width(160)))
                {
                    _controller = DepthTerrainSceneSetup.CreateOrUpdateShowcaseScene();
                }
            }
        }

        private void DrawBundledSamplesSection()
        {
            EditorGUILayout.LabelField("2. Instant Bundled Depth Pro 2 Presets", EditorStyles.boldLabel);
            using (new EditorGUILayout.HorizontalScope())
            {
                if (_bundledSamples.Length == 0)
                {
                    EditorGUILayout.LabelField("No presets found in StreamingAssets/DepthTerrainSamples.");
                }
                else
                {
                    _selectedSampleIdx = EditorGUILayout.Popup("Preset", _selectedSampleIdx, _bundledSamples);
                    if (GUILayout.Button("Load Preset", GUILayout.Width(100)))
                    {
                        EnsureControllerExists();
                        _controller.LoadDataset(_bundledSamples[_selectedSampleIdx], applyRecommendedSettings: true);
                        Repaint();
                    }
                }
                if (GUILayout.Button("Refresh", GUILayout.Width(70)))
                {
                    RefreshSamples();
                }
            }
        }

        private void DrawSourceTabSection()
        {
            EditorGUILayout.LabelField("3. Top-Down Input Source", EditorStyles.boldLabel);
            _request.sourceMode = (PipelineSourceMode)GUILayout.Toolbar((int)_request.sourceMode, TabNames);
            EditorGUILayout.Space(4);

            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                switch (_request.sourceMode)
                {
                    case PipelineSourceMode.RealMap:
                        DrawRealMapTab();
                        break;
                    case PipelineSourceMode.LocalImage:
                        DrawLocalImageTab();
                        break;
                    case PipelineSourceMode.FictionalWorld:
                        DrawFictionalTab();
                        break;
                    case PipelineSourceMode.AerialVideo:
                        DrawVideoTab();
                        break;
                }
            }
        }

        private void DrawRealMapTab()
        {
            EditorGUILayout.LabelField("Quick Famous Locations:", EditorStyles.miniBoldLabel);
            int cols = 2;
            for (int i = 0; i < RealPresets.Length; i += cols)
            {
                using (new EditorGUILayout.HorizontalScope())
                {
                    for (int c = 0; c < cols && (i + c) < RealPresets.Length; c++)
                    {
                        var p = RealPresets[i + c];
                        if (GUILayout.Button(p.Name))
                        {
                            _request.placeQuery = p.Name;
                            _request.useCoordinates = true;
                            _request.latitude = p.Lat;
                            _request.longitude = p.Lon;
                            _request.zoom = p.Zoom;
                            _request.mapProvider = p.Provider;
                            _selectedProviderIdx = Array.IndexOf(DepthPipelineBridge.MapProviders, p.Provider);
                            if (_selectedProviderIdx < 0) _selectedProviderIdx = 0;
                        }
                    }
                }
            }

            EditorGUILayout.Space(4);
            _request.useCoordinates = EditorGUILayout.Toggle("Use Exact Lat / Lon", _request.useCoordinates);
            _request.placeQuery = EditorGUILayout.TextField("Place Name / Label", _request.placeQuery);
            if (_request.useCoordinates)
            {
                _request.latitude = EditorGUILayout.FloatField("Latitude", _request.latitude);
                _request.longitude = EditorGUILayout.FloatField("Longitude", _request.longitude);
            }

            _selectedProviderIdx = EditorGUILayout.Popup("Tile Provider", _selectedProviderIdx, DepthPipelineBridge.MapProviders);
            _request.mapProvider = DepthPipelineBridge.MapProviders[_selectedProviderIdx];
            _request.customTileUrl = EditorGUILayout.TextField("Custom XYZ URL (Optional)", _request.customTileUrl);
            _request.zoom = EditorGUILayout.IntSlider("Map Zoom", _request.zoom, 6, 18);
            _request.tileGrid = EditorGUILayout.IntSlider("Tile Grid (NxN)", _request.tileGrid, 2, 6);
        }

        private void DrawLocalImageTab()
        {
            EditorGUILayout.LabelField("Use any top-down real aerial photo or fictional map image:", EditorStyles.miniLabel);
            using (new EditorGUILayout.HorizontalScope())
            {
                _request.localFilePath = EditorGUILayout.TextField("Image Path", _request.localFilePath);
                if (GUILayout.Button("Browse...", GUILayout.Width(80)))
                {
                    string picked = EditorUtility.OpenFilePanel("Select Top-Down Image", "", "png,jpg,jpeg,webp,heic,tif");
                    if (!string.IsNullOrEmpty(picked)) _request.localFilePath = picked;
                }
            }
        }

        private void DrawFictionalTab()
        {
            EditorGUILayout.LabelField("Synthesize a photorealistic fictional top-down world and infer height with Depth Pro 2:", EditorStyles.miniLabel);
            _selectedBiomeIdx = EditorGUILayout.Popup("Biome", _selectedBiomeIdx, DepthPipelineBridge.FictionalBiomes);
            _request.fictionalBiome = DepthPipelineBridge.FictionalBiomes[_selectedBiomeIdx];
            using (new EditorGUILayout.HorizontalScope())
            {
                _request.fictionalSeed = EditorGUILayout.IntField("Seed", _request.fictionalSeed);
                if (GUILayout.Button("Randomize", GUILayout.Width(90)))
                {
                    _request.fictionalSeed = UnityEngine.Random.Range(1, 99999);
                }
            }
            _request.fictionalImageSize = EditorGUILayout.IntSlider("Image Size (px)", _request.fictionalImageSize, 512, 1024);
        }

        private void DrawVideoTab()
        {
            EditorGUILayout.LabelField("Temporally consistent Depth Pro 2 video inference for aerial/drone clips:", EditorStyles.miniLabel);
            using (new EditorGUILayout.HorizontalScope())
            {
                _request.localFilePath = EditorGUILayout.TextField("Video Path", _request.localFilePath);
                if (GUILayout.Button("Browse...", GUILayout.Width(80)))
                {
                    string picked = EditorUtility.OpenFilePanel("Select Aerial Video", "", "mp4,mov,m4v,webm,mkv");
                    if (!string.IsNullOrEmpty(picked)) _request.localFilePath = picked;
                }
            }
            _request.videoMaxFrames = EditorGUILayout.IntSlider("Max Frames", _request.videoMaxFrames, 4, 48);
            _request.videoStride = EditorGUILayout.IntSlider("Frame Stride", _request.videoStride, 1, 30);

            if (_controller != null && _controller.AvailableVideoFrames > 1)
            {
                EditorGUILayout.Space(4);
                int frame = EditorGUILayout.IntSlider("Video Frame Scrub", _controller.ActiveVideoFrameIndex, 0, _controller.AvailableVideoFrames - 1);
                if (frame != _controller.ActiveVideoFrameIndex)
                {
                    _controller.LoadVideoFrame(frame);
                }
            }
        }

        private void DrawDepthPro2OptionsSection()
        {
            EditorGUILayout.LabelField("4. Apple Depth Pro 2 Inference Settings", EditorStyles.boldLabel);
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                _request.model = (DepthPro2ModelVariant)EditorGUILayout.EnumPopup("Model Variant", _request.model);
                _request.topDownMode = (TopDownAdaptationMode)EditorGUILayout.EnumPopup("Top-Down Mode", _request.topDownMode);
                _request.device = EditorGUILayout.TextField("Compute Device", _request.device);
                _request.heightmapResolution = EditorGUILayout.IntPopup(
                    "Heightmap Resolution",
                    HeightmapProcessor.NearestUnityResolution(_request.heightmapResolution),
                    new[] { "129", "257", "513", "1025" },
                    new[] { 129, 257, 513, 1025 }
                );
                _request.smoothSigma = EditorGUILayout.Slider("Pre-Smooth Sigma", _request.smoothSigma, 0f, 4f);
                _request.useGuidedFilter = EditorGUILayout.Toggle("Guided Filter (Edge-Preserving)", _request.useGuidedFilter);
                _request.exportPointCloudPly = EditorGUILayout.Toggle("Export Metric Point Cloud (.PLY)", _request.exportPointCloudPly);

                EditorGUILayout.Space(4);
                DrawWarmServerBar();
            }
        }

        private void DrawWarmServerBar()
        {
            bool managedRunning = DepthPipelineBridge.IsManagedWarmServerRunning;
            bool healthy = _warmServerHealth != null && _warmServerHealth.IsHealthy;

            string statusText;
            if (healthy)
            {
                statusText = $"Warm Server: ONLINE ({_warmServerHealth.model} | {_warmServerHealth.device} | {_warmServerHealth.uptimeSeconds:F0}s)";
            }
            else if (managedRunning)
            {
                statusText = "Warm Server: STARTING (loading Depth Pro 2 into MPS...)";
            }
            else
            {
                statusText = "Warm Server: OFFLINE (will auto-fallback to uv CLI)";
            }

            EditorGUILayout.LabelField(statusText, EditorStyles.miniBoldLabel);
            using (new EditorGUILayout.HorizontalScope())
            {
                _request.preferWarmServer = EditorGUILayout.ToggleLeft("Prefer Warm Server", _request.preferWarmServer, GUILayout.Width(145));

                if (managedRunning)
                {
                    if (GUILayout.Button("Stop Warm Server", GUILayout.Height(22)))
                    {
                        DepthPipelineBridge.StopWarmServerProcess();
                        _warmServerHealth = new WarmServerHealth { IsHealthy = false };
                        AppendLiveLog("[WarmServer] Stopped managed server process.");
                    }
                }
                else if (!healthy)
                {
                    if (GUILayout.Button("Start Warm Server", GUILayout.Height(22)))
                    {
                        DepthPipelineBridge.StartWarmServerProcess(
                            model: _request.model,
                            device: _request.device,
                            precision: _request.precision,
                            pipelineDirectoryOverride: _controller != null ? _controller.customPipelineDirectory : null,
                            uvExecutableOverride: _controller != null ? _controller.customUvExecutable : null,
                            onLogLine: AppendLiveLog
                        );
                        RefreshWarmServerHealthAsync();
                    }
                }

                using (new EditorGUI.DisabledScope(_checkingServerHealth))
                {
                    if (GUILayout.Button(_checkingServerHealth ? "..." : "Check", GUILayout.Width(58), GUILayout.Height(22)))
                    {
                        RefreshWarmServerHealthAsync();
                    }
                }
            }
        }

        private void DrawGenerateActions()
        {
            bool busy = _controller != null && _controller.IsGenerating;
            using (new EditorGUILayout.HorizontalScope())
            {
                using (new EditorGUI.DisabledScope(busy))
                {
                    if (GUILayout.Button("Generate 3D Terrain with Apple Depth Pro 2", GUILayout.Height(34)))
                    {
                        RunPipelineAsync();
                    }
                }

                if (busy && GUILayout.Button("Cancel", GUILayout.Width(80), GUILayout.Height(34)))
                {
                    _controller.CancelActiveGeneration();
                }
            }

            if (_controller != null)
            {
                EditorGUILayout.LabelField("Status:", _controller.StatusMessage, EditorStyles.miniLabel);
            }
        }

        private void DrawTerrainShapingSection()
        {
            if (_controller == null) return;

            EditorGUILayout.LabelField("5. Real-Time Terrain Shaping (Instant — No Re-Inference)", EditorStyles.boldLabel);
            var s = _controller.buildSettings;
            EditorGUI.BeginChangeCheck();
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                s.terrainSize = EditorGUILayout.Slider("Terrain Size (m)", s.terrainSize, 100f, 2000f);
                s.terrainHeight = EditorGUILayout.Slider("Elevation Height (m)", s.terrainHeight, 10f, 500f);
                s.smoothingPasses = EditorGUILayout.IntSlider("Smoothing Passes", s.smoothingPasses, 0, 10);
                s.elevationGamma = EditorGUILayout.Slider("Elevation Gamma", s.elevationGamma, 0.35f, 2.5f);
                s.invertHeight = EditorGUILayout.Toggle("Invert Depth Polarity", s.invertHeight);
                s.floorClamp = EditorGUILayout.Slider("Floor Clamp", s.floorClamp, 0f, 0.5f);
                s.terraceSteps = EditorGUILayout.IntSlider("Terracing Steps", s.terraceSteps, 0, 24);
                s.cliffDetailStrength = EditorGUILayout.Slider("Cliff Detail Relief", s.cliffDetailStrength, 0f, 1.5f);
                s.normalMapStrength = EditorGUILayout.Slider("Normal Map Strength", s.normalMapStrength, 0f, 3.5f);
                s.useTerrainMasks = EditorGUILayout.Toggle("Use Topographic Masks (AO)", s.useTerrainMasks);
                s.showWaterPlane = EditorGUILayout.Toggle("Show Water Plane", s.showWaterPlane);
                if (s.showWaterPlane)
                {
                    s.waterLevel = EditorGUILayout.Slider("Water Level", s.waterLevel, 0.01f, 0.7f);
                }
                s.edgeFalloff = EditorGUILayout.Toggle("Island Edge Falloff", s.edgeFalloff);
            }

            if (EditorGUI.EndChangeCheck())
            {
                _controller.RebuildTerrainFromCache();
                SceneView.RepaintAll();
            }

            if (_controller.LastBuildTimings.Resolution > 0)
            {
                var t = _controller.LastBuildTimings;
                EditorGUILayout.HelpBox(
                    $"C# Rebuild Telemetry ({t.Resolution}x{t.Resolution}): Total {t.TotalBuildMs:F2} ms " +
                    $"(Process {t.ProcessMs:F2} ms | NormalMap {t.NormalMapMs:F2} ms | Upload {t.TerrainUploadMs:F2} ms | Decode {t.DecodeMs:F2} ms)",
                    MessageType.None
                );
            }

            if (GUILayout.Button("Rebuild Terrain & Frame Camera"))
            {
                _controller.RebuildTerrainFromCache();
                if (_controller.orbitCamera != null)
                    _controller.orbitCamera.FrameTerrain(s.terrainSize, s.terrainHeight, immediate: true);
                SceneView.RepaintAll();
            }
        }

        private void DrawPreviewAndLogsSection()
        {
            if (_controller != null && (_controller.CurrentAlbedoTexture != null || _controller.CurrentDepthPreviewTexture != null))
            {
                EditorGUILayout.LabelField("6. Albedo & Depth Pro 2 Preview", EditorStyles.boldLabel);
                using (new EditorGUILayout.HorizontalScope())
                {
                    float size = Mathf.Min(210f, (position.width - 50f) * 0.5f);
                    if (_controller.CurrentAlbedoTexture != null)
                    {
                        using (new EditorGUILayout.VerticalScope())
                        {
                            EditorGUILayout.LabelField("Top-Down RGB", EditorStyles.centeredGreyMiniLabel, GUILayout.Width(size));
                            Rect r = GUILayoutUtility.GetRect(size, size, GUILayout.Width(size), GUILayout.Height(size));
                            EditorGUI.DrawPreviewTexture(r, _controller.CurrentAlbedoTexture, null, ScaleMode.ScaleToFit);
                        }
                    }

                    if (_controller.CurrentDepthPreviewTexture != null)
                    {
                        using (new EditorGUILayout.VerticalScope())
                        {
                            EditorGUILayout.LabelField("Depth Pro 2 Elevation (Turbo)", EditorStyles.centeredGreyMiniLabel, GUILayout.Width(size));
                            Rect r = GUILayoutUtility.GetRect(size, size, GUILayout.Width(size), GUILayout.Height(size));
                            EditorGUI.DrawPreviewTexture(r, _controller.CurrentDepthPreviewTexture, null, ScaleMode.ScaleToFit);
                        }
                    }
                }
            }

            if (_liveLogs.Length > 0)
            {
                EditorGUILayout.Space(6);
                EditorGUILayout.LabelField("Pipeline Output Log", EditorStyles.miniBoldLabel);
                EditorGUILayout.TextArea(_liveLogs.ToString(), GUILayout.Height(90));
            }
        }

        private void EnsureControllerExists()
        {
            if (_controller == null)
            {
                _controller = FindAnyObjectByType<DepthTerrainController>()
                    ?? DepthTerrainSceneSetup.CreateOrUpdateShowcaseScene(defaultDataset: "", saveScene: false);
            }
        }

        private void AppendLiveLog(string line)
        {
            lock (_liveLogs)
            {
                _liveLogs.AppendLine(line);
            }
            EditorApplication.delayCall += Repaint;
        }

        private async void RefreshWarmServerHealthAsync()
        {
            if (_checkingServerHealth) return;
            _checkingServerHealth = true;
            try
            {
                _warmServerHealth = await DepthPipelineBridge.CheckWarmServerHealthAsync(_request.warmServerUrl, timeoutMs: 300);
            }
            finally
            {
                _checkingServerHealth = false;
                EditorApplication.delayCall += Repaint;
            }
        }

        private async void RunPipelineAsync()
        {
            EnsureControllerExists();
            _liveLogs.Clear();
            _request.outputFolder = Path.Combine(
                Application.persistentDataPath,
                "DepthTerrainOutputs",
                $"{_request.sourceMode}_{DateTime.Now:yyyyMMdd_HHmmss}"
            );

            await _controller.GenerateWithDepthPro2Async(_request, AppendLiveLog);

            RefreshWarmServerHealthAsync();
            SceneView.RepaintAll();
            Repaint();
        }
    }
}
