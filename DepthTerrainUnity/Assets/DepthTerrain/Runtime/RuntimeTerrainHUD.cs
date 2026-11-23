using System;
using UnityEngine;

namespace DepthTerrain
{
    /// <summary>
    /// Interactive Play Mode glass-panel HUD for switching real/fictional Apple Depth Pro 2 presets,
    /// launching live Depth Pro 2 inference for any real place or fictional biome, and tuning 3D terrain parameters.
    /// </summary>
    [RequireComponent(typeof(DepthTerrainController))]
    public class RuntimeTerrainHUD : MonoBehaviour
    {
        public bool showHud = true;
        public bool showPreviews = true;

        private DepthTerrainController _controller;
        private string[] _samples = Array.Empty<string>();
        private Vector2 _scroll;
        private int _modeTab; // 0 = Presets, 1 = Live Real Map, 2 = Live Fictional
        private string _placeInput = "Dolomites, Italy";
        private int _zoom = 14;
        private int _providerIdx;
        private int _biomeIdx;
        private int _seed = 108;
        private int _modelIdx = 1; // 0=S, 1=B, 2=L, 3=H+
        private int _topDownModeIdx; // 0=FastMultiscale, 1=MultiscaleTTA, 2=TTA4, 3=Raw
        private bool _useGuidedFilter;
        private bool _exportPly;
        private bool _preferWarmServer = true;
        private WarmServerHealth _warmServerHealth = new WarmServerHealth();
        private bool _checkingWarmServer;
        private float _sunElevation = 34f;
        private float _sunAzimuth = -42f;

        private GUIStyle _panelStyle;
        private GUIStyle _headerStyle;
        private GUIStyle _sectionStyle;
        private GUIStyle _labelStyle;
        private GUIStyle _buttonStyle;
        private GUIStyle _activeButtonStyle;
        private Texture2D _panelBg;
        private Texture2D _btnBg;
        private Texture2D _btnActiveBg;
        private bool _stylesBuilt;

        private static readonly string[] ModelLabels = { "S (97MB)", "B (386MB)", "L (1.3GB)", "H+ (3.5GB)" };
        private static readonly string[] TopDownModeLabels = { "FastMS", "FullMS", "TTA4", "Raw" };
        private static readonly TopDownAdaptationMode[] TopDownModes =
        {
            TopDownAdaptationMode.FastMultiscale,
            TopDownAdaptationMode.MultiscaleTTA,
            TopDownAdaptationMode.TTA4_Fast,
            TopDownAdaptationMode.RawSinglePass
        };

        private void Awake()
        {
            _controller = GetComponent<DepthTerrainController>();
            _samples = DepthTerrainController.DiscoverBundledSamples();
            RefreshWarmServerStatusAsync();
        }

        private void OnDestroy()
        {
            if (_panelBg != null) Destroy(_panelBg);
            if (_btnBg != null) Destroy(_btnBg);
            if (_btnActiveBg != null) Destroy(_btnActiveBg);
        }

        private void EnsureStyles()
        {
            if (_stylesBuilt) return;
            _stylesBuilt = true;

            _panelBg = MakeSolidTex(new Color(0.08f, 0.10f, 0.14f, 0.86f));
            _btnBg = MakeSolidTex(new Color(0.18f, 0.22f, 0.29f, 0.95f));
            _btnActiveBg = MakeSolidTex(new Color(0.14f, 0.50f, 0.78f, 0.98f));

            _panelStyle = new GUIStyle(GUI.skin.box)
            {
                normal = { background = _panelBg, textColor = Color.white },
                padding = new RectOffset(12, 12, 10, 10)
            };

            _headerStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 15,
                fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.95f, 0.97f, 1f) }
            };

            _sectionStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 12,
                fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.55f, 0.84f, 1f) }
            };

            _labelStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 11,
                wordWrap = true,
                normal = { textColor = new Color(0.86f, 0.89f, 0.94f) }
            };

            _buttonStyle = new GUIStyle(GUI.skin.button)
            {
                fontSize = 11,
                normal = { background = _btnBg, textColor = Color.white },
                hover = { background = _btnActiveBg, textColor = Color.white },
                active = { background = _btnActiveBg, textColor = Color.white }
            };

            _activeButtonStyle = new GUIStyle(_buttonStyle)
            {
                fontStyle = FontStyle.Bold,
                normal = { background = _btnActiveBg, textColor = Color.white }
            };
        }

        private static Texture2D MakeSolidTex(Color c)
        {
            var t = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            t.SetPixel(0, 0, c);
            t.Apply();
            return t;
        }

        private void OnGUI()
        {
            if (!Application.isPlaying) return;
            EnsureStyles();

            const float margin = 14f;
            float panelWidth = showHud ? 345f : 150f;
            float panelHeight = showHud ? Mathf.Min(Screen.height - margin * 2f, 740f) : 38f;
            var panelRect = new Rect(margin, margin, panelWidth, panelHeight);

            if (_controller != null && _controller.orbitCamera != null)
            {
                Vector2 mouseGui = Event.current.mousePosition;
                bool overMain = panelRect.Contains(mouseGui);
                bool overPreview = showHud && showPreviews && GetPreviewRect().Contains(mouseGui);
                _controller.orbitCamera.PointerOverUi = overMain || overPreview;
            }

            GUILayout.BeginArea(panelRect, _panelStyle);

            GUILayout.BeginHorizontal();
            GUILayout.Label("Apple Depth Pro 2 Terrain", _headerStyle);
            if (GUILayout.Button(showHud ? "Hide" : "Show HUD", _buttonStyle, GUILayout.Width(showHud ? 54 : 84)))
            {
                showHud = !showHud;
            }
            GUILayout.EndHorizontal();

            if (!showHud)
            {
                GUILayout.EndArea();
                return;
            }

            GUILayout.Label("Drag: Orbit | Scroll: Zoom | WASD/QE: Move | R: Reset", _labelStyle);
            GUILayout.Space(6);

            _scroll = GUILayout.BeginScrollView(_scroll);

            // Tabs
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Presets", _modeTab == 0 ? _activeButtonStyle : _buttonStyle)) _modeTab = 0;
            if (GUILayout.Button("Live Map", _modeTab == 1 ? _activeButtonStyle : _buttonStyle)) _modeTab = 1;
            if (GUILayout.Button("Fictional", _modeTab == 2 ? _activeButtonStyle : _buttonStyle)) _modeTab = 2;
            GUILayout.EndHorizontal();
            GUILayout.Space(6);

            if (_modeTab == 0) DrawPresetsTab();
            else if (_modeTab == 1) DrawLiveMapTab();
            else DrawLiveFictionalTab();

            GUILayout.Space(8);
            DrawTerrainTuningControls();
            GUILayout.Space(8);
            DrawTelemetrySection();

            GUILayout.EndScrollView();
            GUILayout.EndArea();

            if (showPreviews)
            {
                DrawPreviewOverlay();
            }
        }

        private void DrawPresetsTab()
        {
            GUILayout.Label("Bundled Apple Depth Pro 2 Terrains", _sectionStyle);
            if (_samples.Length == 0)
            {
                GUILayout.Label("No presets found in StreamingAssets/DepthTerrainSamples.", _labelStyle);
                return;
            }

            foreach (string sample in _samples)
            {
                bool isCurrent = string.Equals(_controller.activeDatasetFolder, sample, StringComparison.OrdinalIgnoreCase);
                string pretty = sample.Replace("_", " ").ToUpperInvariant();
                if (GUILayout.Button(pretty, isCurrent ? _activeButtonStyle : _buttonStyle, GUILayout.Height(26)))
                {
                    _controller.LoadDataset(sample, applyRecommendedSettings: true);
                }
            }
        }

        private void DrawLiveMapTab()
        {
            GUILayout.Label("Generate Any Real Place (Maps -> Depth Pro 2)", _sectionStyle);
            GUILayout.Label("Place Name (OSM Nominatim):", _labelStyle);
            _placeInput = GUILayout.TextField(_placeInput);

            GUILayout.Label($"Provider: {DepthPipelineBridge.MapProviders[_providerIdx]}", _labelStyle);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("< Prev", _buttonStyle))
                _providerIdx = (_providerIdx + DepthPipelineBridge.MapProviders.Length - 1) % DepthPipelineBridge.MapProviders.Length;
            if (GUILayout.Button("Next >", _buttonStyle))
                _providerIdx = (_providerIdx + 1) % DepthPipelineBridge.MapProviders.Length;
            GUILayout.EndHorizontal();

            GUILayout.Label($"Zoom Level: {_zoom}", _labelStyle);
            _zoom = Mathf.RoundToInt(GUILayout.HorizontalSlider(_zoom, 10, 17));

            DrawModelSelector();
            DrawInferenceOptions();

            GUI.enabled = !_controller.IsGenerating;
            if (GUILayout.Button(_controller.IsGenerating ? "Running Depth Pro 2..." : "Fetch Map & Generate Terrain", _activeButtonStyle, GUILayout.Height(30)))
            {
                var req = new PipelineRunRequest
                {
                    sourceMode = PipelineSourceMode.RealMap,
                    placeQuery = _placeInput,
                    useCoordinates = false,
                    zoom = _zoom,
                    tileGrid = 3,
                    mapProvider = DepthPipelineBridge.MapProviders[_providerIdx],
                    model = (DepthPro2ModelVariant)_modelIdx,
                    topDownMode = TopDownModes[_topDownModeIdx],
                    useGuidedFilter = _useGuidedFilter,
                    exportPointCloudPly = _exportPly,
                    preferWarmServer = _preferWarmServer
                };
                _ = _controller.GenerateWithDepthPro2Async(req);
            }
            GUI.enabled = true;
        }

        private void DrawLiveFictionalTab()
        {
            GUILayout.Label("Generate Fictional Top-Down World (Depth Pro 2)", _sectionStyle);
            GUILayout.Label($"Biome: {DepthPipelineBridge.FictionalBiomes[_biomeIdx]}", _labelStyle);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("< Prev", _buttonStyle))
                _biomeIdx = (_biomeIdx + DepthPipelineBridge.FictionalBiomes.Length - 1) % DepthPipelineBridge.FictionalBiomes.Length;
            if (GUILayout.Button("Next >", _buttonStyle))
                _biomeIdx = (_biomeIdx + 1) % DepthPipelineBridge.FictionalBiomes.Length;
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            GUILayout.Label($"Seed: {_seed}", _labelStyle);
            if (GUILayout.Button("Random Seed", _buttonStyle, GUILayout.Width(100)))
                _seed = UnityEngine.Random.Range(1, 99999);
            GUILayout.EndHorizontal();

            DrawModelSelector();
            DrawInferenceOptions();

            GUI.enabled = !_controller.IsGenerating;
            if (GUILayout.Button(_controller.IsGenerating ? "Running Depth Pro 2..." : "Synthesize & Infer with Depth Pro 2", _activeButtonStyle, GUILayout.Height(30)))
            {
                var req = new PipelineRunRequest
                {
                    sourceMode = PipelineSourceMode.FictionalWorld,
                    fictionalBiome = DepthPipelineBridge.FictionalBiomes[_biomeIdx],
                    fictionalSeed = _seed,
                    fictionalImageSize = 768,
                    model = (DepthPro2ModelVariant)_modelIdx,
                    topDownMode = TopDownModes[_topDownModeIdx],
                    useGuidedFilter = _useGuidedFilter,
                    exportPointCloudPly = _exportPly,
                    preferWarmServer = _preferWarmServer
                };
                _ = _controller.GenerateWithDepthPro2Async(req);
            }
            GUI.enabled = true;
        }

        private void DrawModelSelector()
        {
            GUILayout.Label("Apple Depth Pro 2 Model:", _labelStyle);
            GUILayout.BeginHorizontal();
            for (int i = 0; i < ModelLabels.Length; i++)
            {
                if (GUILayout.Button(ModelLabels[i], _modelIdx == i ? _activeButtonStyle : _buttonStyle))
                    _modelIdx = i;
            }
            GUILayout.EndHorizontal();
            GUILayout.Space(4);
        }

        private void DrawInferenceOptions()
        {
            GUILayout.Label("Top-Down Mode:", _labelStyle);
            GUILayout.BeginHorizontal();
            for (int i = 0; i < TopDownModeLabels.Length; i++)
            {
                if (GUILayout.Button(TopDownModeLabels[i], _topDownModeIdx == i ? _activeButtonStyle : _buttonStyle))
                    _topDownModeIdx = i;
            }
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            _useGuidedFilter = GUILayout.Toggle(_useGuidedFilter, " Guided Filter");
            _exportPly = GUILayout.Toggle(_exportPly, " Export .PLY");
            GUILayout.EndHorizontal();

            DrawWarmServerControls();
            GUILayout.Space(4);
        }

        private void DrawWarmServerControls()
        {
            bool managedRunning = DepthPipelineBridge.IsManagedWarmServerRunning;
            bool healthy = _warmServerHealth != null && _warmServerHealth.IsHealthy;
            string status = healthy
                ? $"Warm Server: ONLINE ({_warmServerHealth.model}, {_warmServerHealth.device})"
                : managedRunning
                    ? "Warm Server: STARTING..."
                    : "Warm Server: OFFLINE (CLI fallback)";

            GUILayout.Label(status, _labelStyle);
            GUILayout.BeginHorizontal();
            _preferWarmServer = GUILayout.Toggle(_preferWarmServer, " Use Server");

            if (managedRunning)
            {
                if (GUILayout.Button("Stop", _buttonStyle, GUILayout.Width(54)))
                {
                    DepthPipelineBridge.StopWarmServerProcess();
                    _warmServerHealth = new WarmServerHealth { IsHealthy = false };
                }
            }
            else if (!healthy)
            {
                if (GUILayout.Button("Start", _buttonStyle, GUILayout.Width(54)))
                {
                    DepthPipelineBridge.StartWarmServerProcess(
                        model: (DepthPro2ModelVariant)_modelIdx,
                        pipelineDirectoryOverride: _controller.customPipelineDirectory,
                        uvExecutableOverride: _controller.customUvExecutable
                    );
                    RefreshWarmServerStatusAsync();
                }
            }

            GUI.enabled = !_checkingWarmServer;
            if (GUILayout.Button(_checkingWarmServer ? "..." : "Ping", _buttonStyle, GUILayout.Width(48)))
            {
                RefreshWarmServerStatusAsync();
            }
            GUI.enabled = true;
            GUILayout.EndHorizontal();
        }

        private async void RefreshWarmServerStatusAsync()
        {
            if (_checkingWarmServer) return;
            _checkingWarmServer = true;
            try
            {
                _warmServerHealth = await DepthPipelineBridge.CheckWarmServerHealthAsync(timeoutMs: 300);
            }
            finally
            {
                _checkingWarmServer = false;
            }
        }

        private void DrawTerrainTuningControls()
        {
            var s = _controller.buildSettings;
            GUILayout.Label("Real-Time Terrain Shaping", _sectionStyle);

            bool changed = false;

            float newHeight = DrawSlider("Elevation Height", s.terrainHeight, 15f, 350f, "F0", "m", ref changed);
            s.terrainHeight = newHeight;

            int newSmooth = Mathf.RoundToInt(DrawSlider("Smoothing Passes", s.smoothingPasses, 0f, 8f, "F0", "", ref changed));
            s.smoothingPasses = newSmooth;

            float newGamma = DrawSlider("Relief Gamma", s.elevationGamma, 0.4f, 2.2f, "F2", "", ref changed);
            s.elevationGamma = newGamma;

            float newWater = DrawSlider("Water Level", s.waterLevel, 0f, 0.65f, "F2", "", ref changed);
            s.waterLevel = newWater;

            float newNormal = DrawSlider("Normal Relief", s.normalMapStrength, 0f, 3.0f, "F1", "", ref changed);
            s.normalMapStrength = newNormal;

            float newCliff = DrawSlider("Cliff Detail", s.cliffDetailStrength, 0f, 1.5f, "F2", "", ref changed);
            s.cliffDetailStrength = newCliff;

            GUILayout.BeginHorizontal();
            bool inv = GUILayout.Toggle(s.invertHeight, " Invert Depth");
            if (inv != s.invertHeight) { s.invertHeight = inv; changed = true; }

            bool water = GUILayout.Toggle(s.showWaterPlane, " Water Plane");
            if (water != s.showWaterPlane) { s.showWaterPlane = water; changed = true; }
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            bool falloff = GUILayout.Toggle(s.edgeFalloff, " Edge Falloff");
            if (falloff != s.edgeFalloff) { s.edgeFalloff = falloff; changed = true; }

            bool masks = GUILayout.Toggle(s.useTerrainMasks, " Topo Masks");
            if (masks != s.useTerrainMasks) { s.useTerrainMasks = masks; changed = true; }
            GUILayout.EndHorizontal();

            if (changed)
            {
                _controller.RebuildTerrainFromCache();
            }

            if (_controller.sunLight != null)
            {
                GUILayout.Space(4);
                GUILayout.Label("Sun Lighting Angle", _sectionStyle);
                bool sunChanged = false;
                _sunElevation = DrawSlider("Sun Pitch", _sunElevation, 8f, 80f, "F0", "°", ref sunChanged);
                _sunAzimuth = DrawSlider("Sun Azimuth", _sunAzimuth, -180f, 180f, "F0", "°", ref sunChanged);
                if (sunChanged)
                {
                    _controller.sunLight.transform.rotation = Quaternion.Euler(_sunElevation, _sunAzimuth, 0f);
                }
            }
        }

        private float DrawSlider(string label, float value, float min, float max, string fmt, string unit, ref bool changed)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label($"{label}: {value.ToString(fmt)}{unit}", _labelStyle, GUILayout.Width(145));
            float next = GUILayout.HorizontalSlider(value, min, max);
            GUILayout.EndHorizontal();
            if (Mathf.Abs(next - value) > 0.001f) changed = true;
            return next;
        }

        private void DrawTelemetrySection()
        {
            GUILayout.Label("Depth Pro 2 & C# Build Telemetry", _sectionStyle);
            GUILayout.Label(_controller.StatusMessage, _labelStyle);
            var m = _controller.CurrentMetadata;
            var t = _controller.LastBuildTimings;
            if (m != null)
            {
                GUILayout.Label(
                    $"Model: {m.modelName} ({m.device} / {m.precision})\n" +
                    $"Inference: {m.inferenceMs:F0} ms | Focal: {m.focalLengthPx:F0} px\n" +
                    $"Footprint: {m.groundWidthMeters:F0}m x {m.groundHeightMeters:F0}m | Grid: {m.rawResolution}x{m.rawResolution}",
                    _labelStyle
                );
            }
            if (t.Resolution > 0)
            {
                GUILayout.Label(
                    $"C# Rebuild ({t.Resolution}x{t.Resolution}): {t.TotalBuildMs:F2} ms\n" +
                    $"Proc: {t.ProcessMs:F2} ms | Norm: {t.NormalMapMs:F2} ms | Upload: {t.TerrainUploadMs:F2} ms | Dec: {t.DecodeMs:F2} ms",
                    _labelStyle
                );
            }
        }

        private Rect GetPreviewRect()
        {
            const float w = 310f;
            const float h = 175f;
            return new Rect(Screen.width - w - 14f, Screen.height - h - 14f, w, h);
        }

        private void DrawPreviewOverlay()
        {
            if (_controller.CurrentAlbedoTexture == null && _controller.CurrentDepthPreviewTexture == null)
                return;

            GUILayout.BeginArea(GetPreviewRect(), _panelStyle);
            GUILayout.BeginHorizontal();
            if (_controller.CurrentAlbedoTexture != null)
            {
                GUILayout.BeginVertical();
                GUILayout.Label("Top-Down RGB", _labelStyle);
                GUILayout.Box(_controller.CurrentAlbedoTexture, GUILayout.Width(134), GUILayout.Height(134));
                GUILayout.EndVertical();
            }
            if (_controller.CurrentDepthPreviewTexture != null)
            {
                GUILayout.BeginVertical();
                GUILayout.Label("Depth Pro 2 Map", _labelStyle);
                GUILayout.Box(_controller.CurrentDepthPreviewTexture, GUILayout.Width(134), GUILayout.Height(134));
                GUILayout.EndVertical();
            }
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }
    }
}
