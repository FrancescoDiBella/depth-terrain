using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace DepthTerrain
{
    [Serializable]
    public class PipelineRunRequest
    {
        public PipelineSourceMode sourceMode = PipelineSourceMode.RealMap;

        // Real Map options
        public string placeQuery = "Mount Vesuvius, Italy";
        public bool useCoordinates;
        public float latitude = 40.8218f;
        public float longitude = 14.4260f;
        public int zoom = 14;
        public int tileGrid = 3;
        public string mapProvider = "esri_satellite";
        public string customTileUrl = "";

        // Local Image / Video options
        public string localFilePath = "";
        public string videoMode = "buffered";
        public int videoWindow = 8;
        public int videoOverlap = 2;
        public int videoMaxFrames = 16;
        public int videoStride = 5;

        // Fictional World options
        public string fictionalBiome = "volcanic_island";
        public int fictionalSeed = 42;
        public int fictionalImageSize = 768;

        // Apple Depth Pro 2 options
        public DepthPro2ModelVariant model = DepthPro2ModelVariant.Base_B;
        public TopDownAdaptationMode topDownMode = TopDownAdaptationMode.FastMultiscale;
        public string device = "auto";
        public string precision = "";
        public int heightmapResolution = 513;
        public float smoothSigma = 1.0f;
        public float gamma = 1.0f;
        public float percentileLow = 2.0f;
        public float percentileHigh = 98.0f;
        public bool invert;
        public bool useGuidedFilter;
        public bool exportPointCloudPly;
        public bool saveNpz;
        public string outputFolder = "";

        // Warm HTTP Server options
        public bool preferWarmServer = true;
        public string warmServerUrl = DepthPipelineBridge.DefaultWarmServerUrl;
    }

    public class PipelineRunResult
    {
        public bool Success;
        public int ExitCode;
        public bool UsedWarmServer;
        public string OutputFolder;
        public string LogOutput;
        public string ErrorMessage;
        public double ElapsedSeconds;
        public DepthTerrainMetadata Metadata;
    }

    [Serializable]
    public class WarmServerHealth
    {
        public bool IsHealthy;
        public string status = "";
        public string model = "";
        public string device = "";
        public string precision = "";
        public double uptimeSeconds;
        public string[] cachedModels = Array.Empty<string>();
        public int requestsServed;
        public string rawJson = "";
    }

    /// <summary>
    /// Asynchronous bridge invoking Apple Depth Pro 2 via the persistent warm HTTP server when available,
    /// seamlessly falling back to the local Python (uv) CLI via safe ArgumentList execution.
    /// Works in both Unity Editor and standalone macOS/Linux/Windows builds.
    /// </summary>
    public static class DepthPipelineBridge
    {
        public const string DefaultWarmServerHost = "127.0.0.1";
        public const int DefaultWarmServerPort = 8765;
        public const string DefaultWarmServerUrl = "http://127.0.0.1:8765";

        private static readonly HttpClient SharedHttpClient = new HttpClient
        {
            Timeout = Timeout.InfiniteTimeSpan
        };

        private static readonly object ServerLock = new object();
        private static Process _managedServerProcess;
        private static bool _quittingHookRegistered;

        public static readonly string[] MapProviders =
        {
            "esri_satellite",
            "esri_shaded_relief",
            "osm_standard",
            "opentopomap",
            "carto_voyager"
        };

        public static readonly string[] FictionalBiomes =
        {
            "volcanic_island",
            "alpine_peaks",
            "canyon_mesa",
            "fantasy_archipelago"
        };

        public static bool IsManagedWarmServerRunning
        {
            get
            {
                lock (ServerLock)
                {
                    return _managedServerProcess != null && !_managedServerProcess.HasExited;
                }
            }
        }

        [Serializable]
        private class WarmCommonInferRequest
        {
            public string outputDir = "";
            public string model = "depthpro2-b";
            public string device = "auto";
            public string precision = "";
            public string topDownMode = "fast_multiscale";
            public int resolution = 513;
            public bool invert;
            public float smoothSigma = 1.0f;
            public float gamma = 1.0f;
            public float pLow = 2.0f;
            public float pHigh = 98.0f;
            public bool guidedFilter;
            public bool exportPly;
            public bool saveNpz;
        }

        [Serializable]
        private class WarmMapInferRequest : WarmCommonInferRequest
        {
            public string place = "";
            public bool useCoordinates;
            public float latitude;
            public float longitude;
            public int zoom = 14;
            public int grid = 3;
            public string provider = "esri_satellite";
            public string customUrl = "";
        }

        [Serializable]
        private class WarmImageInferRequest : WarmCommonInferRequest
        {
            public string inputPath = "";
            public string label = "";
        }

        [Serializable]
        private class WarmFictionalInferRequest : WarmCommonInferRequest
        {
            public string biome = "volcanic_island";
            public int seed = 42;
            public int size = 768;
        }

        [Serializable]
        private class WarmVideoInferRequest : WarmCommonInferRequest
        {
            public string inputPath = "";
            public string mode = "buffered";
            public int window = 8;
            public int overlap = 2;
            public int maxFrames = 16;
            public int stride = 5;
        }

        [Serializable]
        private class WarmInferResponse
        {
            public string status = "";
            public string outputDir = "";
            public float elapsedMs;
            public DepthTerrainMetadata metadata;
            public string error = "";
        }

        /// <summary>
        /// Locate the sibling `pipeline/` directory containing pyproject.toml.
        /// </summary>
        public static string ResolvePipelineDirectory(string customOverride = null)
        {
            if (!string.IsNullOrWhiteSpace(customOverride) && Directory.Exists(customOverride))
                return Path.GetFullPath(customOverride);

            var candidates = new[]
            {
                Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "pipeline")),
                Path.GetFullPath(Path.Combine(Application.dataPath, "..", "pipeline")),
                Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "pipeline"))
            };

            foreach (string path in candidates)
            {
                if (File.Exists(Path.Combine(path, "pyproject.toml")))
                    return path;
            }

            return candidates[0];
        }

        /// <summary>
        /// Locate the `uv` executable across common macOS/Linux/Windows locations.
        /// </summary>
        public static string ResolveUvExecutable(string customOverride = null)
        {
            if (!string.IsNullOrWhiteSpace(customOverride) && File.Exists(customOverride))
                return customOverride;

            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            var candidates = new[]
            {
                Path.Combine(home, ".local", "bin", "uv"),
                Path.Combine(home, ".cargo", "bin", "uv"),
                "/opt/homebrew/bin/uv",
                "/usr/local/bin/uv",
                "uv"
            };

            foreach (string candidate in candidates)
            {
                if (candidate == "uv" || File.Exists(candidate))
                    return candidate;
            }

            return "uv";
        }

        /// <summary>
        /// Build safe CLI argument list for `uv run depth-terrain ...`.
        /// </summary>
        public static List<string> BuildArguments(PipelineRunRequest req, string outputFolder)
        {
            var inv = CultureInfo.InvariantCulture;
            var args = new List<string> { "run", "depth-terrain" };

            switch (req.sourceMode)
            {
                case PipelineSourceMode.RealMap:
                    args.Add("from-map");
                    if (req.useCoordinates)
                    {
                        args.Add("--lat");
                        args.Add(req.latitude.ToString("F6", inv));
                        args.Add("--lon");
                        args.Add(req.longitude.ToString("F6", inv));
                        if (!string.IsNullOrWhiteSpace(req.placeQuery))
                        {
                            args.Add("--place");
                            args.Add(req.placeQuery.Trim());
                        }
                    }
                    else
                    {
                        args.Add("--place");
                        args.Add(string.IsNullOrWhiteSpace(req.placeQuery) ? "Mount Vesuvius, Italy" : req.placeQuery.Trim());
                    }

                    args.Add("--zoom");
                    args.Add(Mathf.Clamp(req.zoom, 1, 19).ToString(inv));
                    args.Add("--grid");
                    args.Add(Mathf.Clamp(req.tileGrid, 1, 8).ToString(inv));

                    if (!string.IsNullOrWhiteSpace(req.customTileUrl))
                    {
                        args.Add("--custom-url");
                        args.Add(req.customTileUrl.Trim());
                    }
                    else
                    {
                        args.Add("--provider");
                        args.Add(string.IsNullOrWhiteSpace(req.mapProvider) ? "esri_satellite" : req.mapProvider);
                    }
                    break;

                case PipelineSourceMode.LocalImage:
                    args.Add("from-image");
                    args.Add("--input");
                    args.Add(req.localFilePath);
                    break;

                case PipelineSourceMode.FictionalWorld:
                    args.Add("from-fictional");
                    args.Add("--biome");
                    args.Add(string.IsNullOrWhiteSpace(req.fictionalBiome) ? "volcanic_island" : req.fictionalBiome);
                    args.Add("--seed");
                    args.Add(req.fictionalSeed.ToString(inv));
                    args.Add("--size");
                    args.Add(Mathf.Clamp(req.fictionalImageSize, 256, 1536).ToString(inv));
                    break;

                case PipelineSourceMode.AerialVideo:
                    args.Add("from-video");
                    args.Add("--input");
                    args.Add(req.localFilePath);
                    args.Add("--mode");
                    args.Add(string.IsNullOrWhiteSpace(req.videoMode) ? "buffered" : req.videoMode);
                    args.Add("--window");
                    args.Add(Mathf.Clamp(req.videoWindow, 2, 32).ToString(inv));
                    args.Add("--overlap");
                    args.Add(Mathf.Clamp(req.videoOverlap, 0, 16).ToString(inv));
                    args.Add("--max-frames");
                    args.Add(Mathf.Clamp(req.videoMaxFrames, 2, 64).ToString(inv));
                    args.Add("--stride");
                    args.Add(Mathf.Max(1, req.videoStride).ToString(inv));
                    break;
            }

            args.Add("--model");
            args.Add(req.model.ToCliFlag());
            args.Add("--device");
            args.Add(string.IsNullOrWhiteSpace(req.device) ? "auto" : req.device);
            if (!string.IsNullOrWhiteSpace(req.precision))
            {
                args.Add("--precision");
                args.Add(req.precision.Trim());
            }
            args.Add("--topdown-mode");
            args.Add(req.topDownMode.ToCliFlag());
            args.Add("--resolution");
            args.Add(HeightmapProcessor.NearestUnityResolution(req.heightmapResolution).ToString(inv));
            args.Add("--smooth-sigma");
            args.Add(req.smoothSigma.ToString("F2", inv));
            args.Add("--gamma");
            args.Add(req.gamma.ToString("F2", inv));
            args.Add("--p-low");
            args.Add(req.percentileLow.ToString("F2", inv));
            args.Add("--p-high");
            args.Add(req.percentileHigh.ToString("F2", inv));
            if (req.invert) args.Add("--invert");
            args.Add(req.useGuidedFilter ? "--guided-filter" : "--no-guided-filter");
            if (req.exportPointCloudPly) args.Add("--export-ply");
            if (req.saveNpz) args.Add("--save-npz");
            args.Add("--output");
            args.Add(outputFolder);

            return args;
        }

        /// <summary>
        /// Resolve the warm HTTP server endpoint URL for a given pipeline source mode.
        /// </summary>
        public static string BuildWarmServerEndpoint(PipelineSourceMode sourceMode, string serverUrl = DefaultWarmServerUrl)
        {
            string baseUrl = NormalizeServerUrl(serverUrl);
            string route = sourceMode switch
            {
                PipelineSourceMode.RealMap => "/infer/map",
                PipelineSourceMode.LocalImage => "/infer/image",
                PipelineSourceMode.FictionalWorld => "/infer/fictional",
                PipelineSourceMode.AerialVideo => "/infer/video",
                _ => "/infer/map"
            };
            return baseUrl + route;
        }

        /// <summary>
        /// Serialize a PipelineRunRequest into the exact camelCase JSON payload expected by `depth-terrain serve`.
        /// </summary>
        public static string BuildWarmServerRequestJson(PipelineRunRequest req, string outputFolder)
        {
            switch (req.sourceMode)
            {
                case PipelineSourceMode.RealMap:
                {
                    var dto = new WarmMapInferRequest
                    {
                        place = string.IsNullOrWhiteSpace(req.placeQuery)
                            ? (req.useCoordinates ? "" : "Mount Vesuvius, Italy")
                            : req.placeQuery.Trim(),
                        useCoordinates = req.useCoordinates,
                        latitude = req.latitude,
                        longitude = req.longitude,
                        zoom = Mathf.Clamp(req.zoom, 1, 19),
                        grid = Mathf.Clamp(req.tileGrid, 1, 8),
                        provider = string.IsNullOrWhiteSpace(req.mapProvider) ? "esri_satellite" : req.mapProvider,
                        customUrl = string.IsNullOrWhiteSpace(req.customTileUrl) ? "" : req.customTileUrl.Trim()
                    };
                    PopulateCommonWarmFields(dto, req, outputFolder);
                    return JsonUtility.ToJson(dto);
                }

                case PipelineSourceMode.LocalImage:
                {
                    var dto = new WarmImageInferRequest
                    {
                        inputPath = req.localFilePath ?? "",
                        label = string.IsNullOrWhiteSpace(req.localFilePath)
                            ? ""
                            : Path.GetFileNameWithoutExtension(req.localFilePath)
                    };
                    PopulateCommonWarmFields(dto, req, outputFolder);
                    return JsonUtility.ToJson(dto);
                }

                case PipelineSourceMode.FictionalWorld:
                {
                    var dto = new WarmFictionalInferRequest
                    {
                        biome = string.IsNullOrWhiteSpace(req.fictionalBiome) ? "volcanic_island" : req.fictionalBiome,
                        seed = req.fictionalSeed,
                        size = Mathf.Clamp(req.fictionalImageSize, 256, 1536)
                    };
                    PopulateCommonWarmFields(dto, req, outputFolder);
                    return JsonUtility.ToJson(dto);
                }

                case PipelineSourceMode.AerialVideo:
                {
                    var dto = new WarmVideoInferRequest
                    {
                        inputPath = req.localFilePath ?? "",
                        mode = string.IsNullOrWhiteSpace(req.videoMode) ? "buffered" : req.videoMode,
                        window = Mathf.Clamp(req.videoWindow, 2, 32),
                        overlap = Mathf.Clamp(req.videoOverlap, 0, 16),
                        maxFrames = Mathf.Clamp(req.videoMaxFrames, 2, 64),
                        stride = Mathf.Max(1, req.videoStride)
                    };
                    PopulateCommonWarmFields(dto, req, outputFolder);
                    return JsonUtility.ToJson(dto);
                }

                default:
                    throw new ArgumentOutOfRangeException(nameof(req.sourceMode));
            }
        }

        private static void PopulateCommonWarmFields(WarmCommonInferRequest dto, PipelineRunRequest req, string outputFolder)
        {
            dto.outputDir = outputFolder ?? "";
            dto.model = req.model.ToCliFlag();
            dto.device = string.IsNullOrWhiteSpace(req.device) ? "auto" : req.device;
            dto.precision = string.IsNullOrWhiteSpace(req.precision) ? "" : req.precision.Trim();
            dto.topDownMode = req.topDownMode.ToCliFlag();
            dto.resolution = HeightmapProcessor.NearestUnityResolution(req.heightmapResolution);
            dto.invert = req.invert;
            dto.smoothSigma = req.smoothSigma;
            dto.gamma = req.gamma;
            dto.pLow = req.percentileLow;
            dto.pHigh = req.percentileHigh;
            dto.guidedFilter = req.useGuidedFilter;
            dto.exportPly = req.exportPointCloudPly;
            dto.saveNpz = req.saveNpz;
        }

        /// <summary>
        /// Probe `GET /health` on the local warm Depth Pro 2 server with a short timeout (default 300ms).
        /// Never throws when the server is offline.
        /// </summary>
        public static async Task<WarmServerHealth> CheckWarmServerHealthAsync(
            string serverUrl = DefaultWarmServerUrl,
            int timeoutMs = 300,
            CancellationToken cancellationToken = default
        )
        {
            string url = NormalizeServerUrl(serverUrl) + "/health";
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            linkedCts.CancelAfter(Math.Max(50, timeoutMs));

            try
            {
                using HttpResponseMessage response = await SharedHttpClient.GetAsync(url, linkedCts.Token).ConfigureAwait(false);
                string body = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
                if (!response.IsSuccessStatusCode || string.IsNullOrWhiteSpace(body))
                {
                    return new WarmServerHealth { IsHealthy = false, rawJson = body ?? "" };
                }

                var health = JsonUtility.FromJson<WarmServerHealth>(body) ?? new WarmServerHealth();
                health.rawJson = body;
                health.IsHealthy = string.Equals(health.status, "ok", StringComparison.OrdinalIgnoreCase);
                return health;
            }
            catch
            {
                return new WarmServerHealth { IsHealthy = false };
            }
        }

        /// <summary>
        /// Launch `uv run depth-terrain serve` as a managed background process so repeated requests
        /// keep Apple Depth Pro 2 weights warm in Apple Silicon MPS / GPU memory.
        /// </summary>
        public static bool StartWarmServerProcess(
            DepthPro2ModelVariant model = DepthPro2ModelVariant.Base_B,
            string host = DefaultWarmServerHost,
            int port = DefaultWarmServerPort,
            string device = "auto",
            string precision = "",
            string pipelineDirectoryOverride = null,
            string uvExecutableOverride = null,
            Action<string> onLogLine = null
        )
        {
            lock (ServerLock)
            {
                if (_managedServerProcess != null && !_managedServerProcess.HasExited)
                {
                    onLogLine?.Invoke($"[WarmServer] Already running (PID {_managedServerProcess.Id}).");
                    return true;
                }

                string pipelineDir = ResolvePipelineDirectory(pipelineDirectoryOverride);
                if (!Directory.Exists(pipelineDir))
                {
                    onLogLine?.Invoke($"[WarmServer] Pipeline directory not found: '{pipelineDir}'.");
                    return false;
                }

                string uvExe = ResolveUvExecutable(uvExecutableOverride);
                var psi = CreateUvProcessStartInfo(uvExe, pipelineDir);

                psi.ArgumentList.Add("run");
                psi.ArgumentList.Add("depth-terrain");
                psi.ArgumentList.Add("serve");
                psi.ArgumentList.Add("--host");
                psi.ArgumentList.Add(string.IsNullOrWhiteSpace(host) ? DefaultWarmServerHost : host.Trim());
                psi.ArgumentList.Add("--port");
                psi.ArgumentList.Add(Mathf.Clamp(port, 1024, 65535).ToString(CultureInfo.InvariantCulture));
                psi.ArgumentList.Add("--model");
                psi.ArgumentList.Add(model.ToCliFlag());
                psi.ArgumentList.Add("--device");
                psi.ArgumentList.Add(string.IsNullOrWhiteSpace(device) ? "auto" : device.Trim());
                if (!string.IsNullOrWhiteSpace(precision))
                {
                    psi.ArgumentList.Add("--precision");
                    psi.ArgumentList.Add(precision.Trim());
                }

                var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
                proc.OutputDataReceived += (_, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data)) onLogLine?.Invoke($"[WarmServer] {e.Data}");
                };
                proc.ErrorDataReceived += (_, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data)) onLogLine?.Invoke($"[WarmServer] {e.Data}");
                };

                try
                {
                    proc.Start();
                    proc.BeginOutputReadLine();
                    proc.BeginErrorReadLine();
                    _managedServerProcess = proc;

                    if (!_quittingHookRegistered)
                    {
                        _quittingHookRegistered = true;
                        Application.quitting += StopWarmServerProcess;
                    }

                    onLogLine?.Invoke($"[WarmServer] Started PID {proc.Id} on http://{host}:{port} ({model.ToCliFlag()}).");
                    return true;
                }
                catch (Exception ex)
                {
                    proc.Dispose();
                    onLogLine?.Invoke($"[WarmServer] Failed to start: {ex.Message}");
                    return false;
                }
            }
        }

        /// <summary>
        /// Cleanly terminate the managed warm Depth Pro 2 server process if started by Unity.
        /// </summary>
        public static void StopWarmServerProcess()
        {
            lock (ServerLock)
            {
                if (_managedServerProcess == null)
                    return;

                try
                {
                    if (!_managedServerProcess.HasExited)
                    {
                        _managedServerProcess.Kill();
                        _managedServerProcess.WaitForExit(1500);
                    }
                }
                catch
                {
                    // Ignore race when process exits concurrently
                }
                finally
                {
                    try { _managedServerProcess.Dispose(); } catch { /* ignore */ }
                    _managedServerProcess = null;
                }
            }
        }

        /// <summary>
        /// Execute the Apple Depth Pro 2 pipeline asynchronously without blocking the Unity main thread.
        /// Automatically probes the warm HTTP server (300ms timeout) when <see cref="PipelineRunRequest.preferWarmServer"/>
        /// is enabled, and seamlessly falls back to `uv run depth-terrain` subprocess if offline.
        /// </summary>
        public static async Task<PipelineRunResult> RunAsync(
            PipelineRunRequest request,
            Action<string> onLogLine = null,
            string pipelineDirectoryOverride = null,
            string uvExecutableOverride = null,
            CancellationToken cancellationToken = default
        )
        {
            string pipelineDir = ResolvePipelineDirectory(pipelineDirectoryOverride);
            string uvExe = ResolveUvExecutable(uvExecutableOverride);

            string outputDir = string.IsNullOrWhiteSpace(request.outputFolder)
                ? Path.Combine(Application.persistentDataPath, "DepthTerrainOutputs", $"run_{DateTime.Now:yyyyMMdd_HHmmss}")
                : request.outputFolder;

            Directory.CreateDirectory(outputDir);

            // 1) Fast path: probe warm Depth Pro 2 HTTP server
            if (request.preferWarmServer)
            {
                string serverUrl = NormalizeServerUrl(request.warmServerUrl);
                string requestJson = BuildWarmServerRequestJson(request, outputDir);
                string endpoint = BuildWarmServerEndpoint(request.sourceMode, serverUrl);

                WarmServerHealth health = await CheckWarmServerHealthAsync(serverUrl, timeoutMs: 300, cancellationToken);
                if (health.IsHealthy)
                {
                    onLogLine?.Invoke(
                        $"[WarmServer] Online ({health.model} on {health.device}, uptime {health.uptimeSeconds:F0}s) -> POST {endpoint}"
                    );
                    var warmResult = await TryRunViaWarmServerAsync(endpoint, requestJson, outputDir, onLogLine, cancellationToken);
                    if (warmResult != null)
                        return warmResult;

                    onLogLine?.Invoke("[WarmServer] Unreachable during request; falling back to uv CLI subprocess...");
                }
            }

            // 2) Fallback path: local `uv run depth-terrain` CLI subprocess
            return await RunViaCliSubprocessAsync(request, outputDir, pipelineDir, uvExe, onLogLine, cancellationToken);
        }

        private static async Task<PipelineRunResult> TryRunViaWarmServerAsync(
            string endpoint,
            string requestJson,
            string outputDir,
            Action<string> onLogLine,
            CancellationToken cancellationToken
        )
        {
            var sw = Stopwatch.StartNew();
            var logBuilder = new StringBuilder();
            logBuilder.AppendLine($"POST {endpoint}");

            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeoutCts.CancelAfter(TimeSpan.FromMinutes(10));

            try
            {
                using var content = new StringContent(requestJson, Encoding.UTF8, "application/json");
                using HttpResponseMessage response = await SharedHttpClient
                    .PostAsync(endpoint, content, timeoutCts.Token)
                    .ConfigureAwait(false);

                string body = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
                sw.Stop();
                logBuilder.AppendLine($"HTTP {(int)response.StatusCode} ({sw.Elapsed.TotalMilliseconds:F1} ms)");

                if (response.StatusCode == HttpStatusCode.NotFound ||
                    response.StatusCode == HttpStatusCode.BadGateway ||
                    response.StatusCode == HttpStatusCode.ServiceUnavailable)
                {
                    return null; // Trigger seamless CLI fallback
                }

                WarmInferResponse parsed = null;
                if (!string.IsNullOrWhiteSpace(body))
                {
                    try { parsed = JsonUtility.FromJson<WarmInferResponse>(body); }
                    catch { /* handled below */ }
                }

                if (!response.IsSuccessStatusCode ||
                    parsed == null ||
                    !string.Equals(parsed.status, "ok", StringComparison.OrdinalIgnoreCase))
                {
                    string err = parsed != null && !string.IsNullOrWhiteSpace(parsed.error)
                        ? parsed.error
                        : $"Warm server returned HTTP {(int)response.StatusCode}: {body}";
                    onLogLine?.Invoke($"[WarmServer] Error: {err}");
                    logBuilder.AppendLine(err);
                    return new PipelineRunResult
                    {
                        Success = false,
                        ExitCode = (int)response.StatusCode,
                        UsedWarmServer = true,
                        OutputFolder = outputDir,
                        LogOutput = logBuilder.ToString(),
                        ErrorMessage = err,
                        ElapsedSeconds = sw.Elapsed.TotalSeconds
                    };
                }

                string resolvedOut = string.IsNullOrWhiteSpace(parsed.outputDir) ? outputDir : parsed.outputDir;
                DepthTerrainMetadata meta = parsed.metadata;
                string metaPath = Path.Combine(resolvedOut, "metadata.json");
                if ((meta == null || string.IsNullOrEmpty(meta.heightmapRawFile)) && File.Exists(metaPath))
                {
                    meta = JsonUtility.FromJson<DepthTerrainMetadata>(File.ReadAllText(metaPath));
                }

                onLogLine?.Invoke($"[WarmServer] Completed in {parsed.elapsedMs:F1} ms -> {resolvedOut}");
                logBuilder.AppendLine($"Completed in {parsed.elapsedMs:F1} ms -> {resolvedOut}");

                return new PipelineRunResult
                {
                    Success = meta != null,
                    ExitCode = 0,
                    UsedWarmServer = true,
                    OutputFolder = resolvedOut,
                    LogOutput = logBuilder.ToString(),
                    ErrorMessage = meta != null ? null : "Warm server response missing metadata.",
                    ElapsedSeconds = sw.Elapsed.TotalSeconds,
                    Metadata = meta
                };
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                sw.Stop();
                return new PipelineRunResult
                {
                    Success = false,
                    ExitCode = -2,
                    UsedWarmServer = true,
                    OutputFolder = outputDir,
                    LogOutput = logBuilder.ToString(),
                    ErrorMessage = "Depth Pro 2 generation cancelled by user.",
                    ElapsedSeconds = sw.Elapsed.TotalSeconds
                };
            }
            catch (HttpRequestException)
            {
                return null; // Trigger seamless CLI fallback
            }
        }

        private static async Task<PipelineRunResult> RunViaCliSubprocessAsync(
            PipelineRunRequest request,
            string outputDir,
            string pipelineDir,
            string uvExe,
            Action<string> onLogLine,
            CancellationToken cancellationToken
        )
        {
            List<string> args = BuildArguments(request, outputDir);
            var sw = Stopwatch.StartNew();
            var logBuilder = new StringBuilder();

            return await Task.Run(() =>
            {
                if (!Directory.Exists(pipelineDir))
                {
                    return new PipelineRunResult
                    {
                        Success = false,
                        ExitCode = -1,
                        UsedWarmServer = false,
                        OutputFolder = outputDir,
                        ErrorMessage = $"Pipeline directory not found at '{pipelineDir}'."
                    };
                }

                var psi = CreateUvProcessStartInfo(uvExe, pipelineDir);
                foreach (string arg in args)
                    psi.ArgumentList.Add(arg);

                using var proc = new Process { StartInfo = psi };
                proc.OutputDataReceived += (_, e) =>
                {
                    if (string.IsNullOrEmpty(e.Data)) return;
                    lock (logBuilder) logBuilder.AppendLine(e.Data);
                    onLogLine?.Invoke(e.Data);
                };
                proc.ErrorDataReceived += (_, e) =>
                {
                    if (string.IsNullOrEmpty(e.Data)) return;
                    lock (logBuilder) logBuilder.AppendLine(e.Data);
                    onLogLine?.Invoke(e.Data);
                };

                try
                {
                    onLogLine?.Invoke($"Launching: {uvExe} {string.Join(" ", args)}");
                    proc.Start();
                    proc.BeginOutputReadLine();
                    proc.BeginErrorReadLine();

                    while (!proc.WaitForExit(150))
                    {
                        if (cancellationToken.IsCancellationRequested)
                        {
                            try { proc.Kill(); } catch { /* ignore */ }
                            return new PipelineRunResult
                            {
                                Success = false,
                                ExitCode = -2,
                                UsedWarmServer = false,
                                OutputFolder = outputDir,
                                LogOutput = logBuilder.ToString(),
                                ErrorMessage = "Depth Pro 2 generation cancelled by user.",
                                ElapsedSeconds = sw.Elapsed.TotalSeconds
                            };
                        }
                    }

                    proc.WaitForExit();
                    sw.Stop();

                    DepthTerrainMetadata meta = null;
                    string metaPath = Path.Combine(outputDir, "metadata.json");
                    if (proc.ExitCode == 0 && File.Exists(metaPath))
                    {
                        meta = JsonUtility.FromJson<DepthTerrainMetadata>(File.ReadAllText(metaPath));
                    }

                    return new PipelineRunResult
                    {
                        Success = proc.ExitCode == 0 && meta != null,
                        ExitCode = proc.ExitCode,
                        UsedWarmServer = false,
                        OutputFolder = outputDir,
                        LogOutput = logBuilder.ToString(),
                        ErrorMessage = proc.ExitCode == 0 ? null : $"Pipeline exited with code {proc.ExitCode}.",
                        ElapsedSeconds = sw.Elapsed.TotalSeconds,
                        Metadata = meta
                    };
                }
                catch (Exception ex)
                {
                    sw.Stop();
                    return new PipelineRunResult
                    {
                        Success = false,
                        ExitCode = -1,
                        UsedWarmServer = false,
                        OutputFolder = outputDir,
                        LogOutput = logBuilder.ToString(),
                        ErrorMessage = ex.Message,
                        ElapsedSeconds = sw.Elapsed.TotalSeconds
                    };
                }
            }, cancellationToken);
        }

        private static ProcessStartInfo CreateUvProcessStartInfo(string uvExe, string pipelineDir)
        {
            var psi = new ProcessStartInfo
            {
                FileName = uvExe,
                WorkingDirectory = pipelineDir,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            // Ensure homebrew/local bins are visible inside GUI-launched Unity Editor on macOS
            string existingPath = Environment.GetEnvironmentVariable("PATH") ?? "";
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            psi.Environment["PATH"] = $"{home}/.local/bin:/opt/homebrew/bin:/usr/local/bin:{existingPath}";
            psi.Environment["PYTHONUNBUFFERED"] = "1";
            return psi;
        }

        private static string NormalizeServerUrl(string serverUrl)
        {
            string url = string.IsNullOrWhiteSpace(serverUrl) ? DefaultWarmServerUrl : serverUrl.Trim();
            return url.TrimEnd('/');
        }
    }
}
