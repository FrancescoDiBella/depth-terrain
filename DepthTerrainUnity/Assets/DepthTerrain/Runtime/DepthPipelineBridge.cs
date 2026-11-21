using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
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
        public int videoMaxFrames = 16;
        public int videoStride = 5;

        // Fictional World options
        public string fictionalBiome = "volcanic_island";
        public int fictionalSeed = 42;
        public int fictionalImageSize = 768;

        // Apple Depth Pro 2 options
        public DepthPro2ModelVariant model = DepthPro2ModelVariant.Base_B;
        public TopDownAdaptationMode topDownMode = TopDownAdaptationMode.MultiscaleTTA;
        public string device = "auto";
        public int heightmapResolution = 513;
        public float smoothSigma = 1.0f;
        public float gamma = 1.0f;
        public bool invert;
        public string outputFolder = "";
    }

    public class PipelineRunResult
    {
        public bool Success;
        public int ExitCode;
        public string OutputFolder;
        public string LogOutput;
        public string ErrorMessage;
        public double ElapsedSeconds;
        public DepthTerrainMetadata Metadata;
    }

    /// <summary>
    /// Asynchronous bridge invoking the local Python (uv) Apple Depth Pro 2 CLI safely via ArgumentList.
    /// Works in both Unity Editor and standalone macOS/Linux/Windows builds.
    /// </summary>
    public static class DepthPipelineBridge
    {
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
            args.Add("--topdown-mode");
            args.Add(req.topDownMode.ToCliFlag());
            args.Add("--resolution");
            args.Add(HeightmapProcessor.NearestUnityResolution(req.heightmapResolution).ToString(inv));
            args.Add("--smooth-sigma");
            args.Add(req.smoothSigma.ToString("F2", inv));
            args.Add("--gamma");
            args.Add(req.gamma.ToString("F2", inv));
            if (req.invert) args.Add("--invert");
            args.Add("--output");
            args.Add(outputFolder);

            return args;
        }

        /// <summary>
        /// Execute the Apple Depth Pro 2 pipeline asynchronously without blocking the Unity main thread.
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
                        OutputFolder = outputDir,
                        ErrorMessage = $"Pipeline directory not found at '{pipelineDir}'."
                    };
                }

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
                        OutputFolder = outputDir,
                        LogOutput = logBuilder.ToString(),
                        ErrorMessage = ex.Message,
                        ElapsedSeconds = sw.Elapsed.TotalSeconds
                    };
                }
            }, cancellationToken);
        }
    }
}
