using System;
using UnityEngine;

namespace DepthTerrain
{
    /// <summary>
    /// JSON contract emitted by the Python Apple Depth Pro 2 pipeline (metadata.json).
    /// </summary>
    [Serializable]
    public class DepthTerrainMetadata
    {
        public int schemaVersion = 1;
        public string sourceType = "";
        public string sourceLabel = "";
        public string modelName = "depthpro2-b";
        public string device = "mps";
        public string precision = "bfloat16";
        public float inferenceMs;
        public float focalLengthPx;
        public int imageWidth;
        public int imageHeight;
        public int rawResolution = 513;
        public string rawByteOrder = "little";
        public bool rawFlipY = true;
        public string albedoFile = "albedo.png";
        public string heightmapRawFile = "heightmap.raw";
        public string heightmapPng16File = "heightmap_16bit.png";
        public string depthPreviewFile = "depth_preview.png";
        public float groundWidthMeters = 500f;
        public float groundHeightMeters = 500f;
        public float recommendedTerrainSize = 500f;
        public float recommendedTerrainHeight = 120f;
        public float recommendedWaterLevel = 0.08f;
        public float latitude;
        public float longitude;
        public int zoom;
        public string mapProvider = "";
        public string biome = "";
        public int seed;
        public int videoFrameCount;
        public float depthMinMeters;
        public float depthMaxMeters;
        public float depthP02Meters;
        public float depthP50Meters;
        public float depthP98Meters;
        public float heightMin;
        public float heightMax;
        public float heightMean;
    }

    /// <summary>
    /// Non-destructive terrain shaping settings applied in real time on top of Depth Pro 2 heightmaps.
    /// </summary>
    [Serializable]
    public class TerrainBuildSettings
    {
        [Tooltip("World width & length of the Unity Terrain in meters/units.")]
        [Range(50f, 4000f)]
        public float terrainSize = 500f;

        [Tooltip("Maximum vertical elevation scale of the Unity Terrain.")]
        [Range(5f, 800f)]
        public float terrainHeight = 125f;

        [Tooltip("Target Unity TerrainData heightmap resolution (snapped to 2^n + 1).")]
        public int targetResolution = 513;

        [Tooltip("Flip height polarity (near <-> far).")]
        public bool invertHeight;

        [Tooltip("Number of separable 3x3 Gaussian smoothing passes (0 = raw Depth Pro 2 detail).")]
        [Range(0, 12)]
        public int smoothingPasses = 1;

        [Tooltip("Contrast/elevation curve exponent (< 1 boosts midtones/plateaus, > 1 sharpens peaks).")]
        [Range(0.3f, 3.0f)]
        public float elevationGamma = 1.0f;

        [Tooltip("Optional sea-level flattening floor [0..0.6].")]
        [Range(0f, 0.6f)]
        public float floorClamp;

        [Tooltip("Optional elevation terracing step count (0 = disabled).")]
        [Range(0, 32)]
        public int terraceSteps;

        [Tooltip("Show translucent URP water plane at waterLevel * terrainHeight.")]
        public bool showWaterPlane = true;

        [Tooltip("Normalized water plane height [0..0.75] relative to terrainHeight.")]
        [Range(0f, 0.75f)]
        public float waterLevel = 0.08f;

        [Tooltip("Strength of procedural normal map baked from the Depth Pro 2 heightmap.")]
        [Range(0f, 4f)]
        public float normalMapStrength = 1.4f;

        [Tooltip("Smoothly fall off terrain edges to zero for clean diorama/island presentation.")]
        public bool edgeFalloff;

        [Range(0.02f, 0.35f)]
        public float edgeFalloffMargin = 0.08f;

        public TerrainBuildSettings Clone()
        {
            return (TerrainBuildSettings)MemberwiseClone();
        }
    }

    public enum PipelineSourceMode
    {
        RealMap = 0,
        LocalImage = 1,
        FictionalWorld = 2,
        AerialVideo = 3
    }

    public enum DepthPro2ModelVariant
    {
        Small_S = 0,
        Base_B = 1,
        Large_L = 2,
        HugePlus_H = 3
    }

    public enum TopDownAdaptationMode
    {
        MultiscaleTTA = 0,
        TTA4_Fast = 1,
        RawSinglePass = 2
    }

    public static class DepthPro2ModelExtensions
    {
        public static string ToCliFlag(this DepthPro2ModelVariant variant) => variant switch
        {
            DepthPro2ModelVariant.Small_S => "depthpro2-s",
            DepthPro2ModelVariant.Base_B => "depthpro2-b",
            DepthPro2ModelVariant.Large_L => "depthpro2-l",
            DepthPro2ModelVariant.HugePlus_H => "depthpro2-h",
            _ => "depthpro2-b"
        };

        public static string ToCliFlag(this TopDownAdaptationMode mode) => mode switch
        {
            TopDownAdaptationMode.MultiscaleTTA => "multiscale_tta",
            TopDownAdaptationMode.TTA4_Fast => "tta4",
            TopDownAdaptationMode.RawSinglePass => "raw",
            _ => "multiscale_tta"
        };
    }
}
