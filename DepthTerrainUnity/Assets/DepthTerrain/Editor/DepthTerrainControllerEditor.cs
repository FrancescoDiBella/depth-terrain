using UnityEditor;
using UnityEngine;

namespace DepthTerrain.Editor
{
    [CustomEditor(typeof(DepthTerrainController))]
    public class DepthTerrainControllerEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            var ctrl = (DepthTerrainController)target;
            EditorGUILayout.Space(8);
            EditorGUILayout.LabelField("Apple Depth Pro 2 Controls", EditorStyles.boldLabel);

            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Load Active Dataset"))
                {
                    ctrl.LoadDataset(ctrl.activeDatasetFolder, applyRecommendedSettings: true);
                    SceneView.RepaintAll();
                }

                if (GUILayout.Button("Rebuild (Instant)"))
                {
                    ctrl.RebuildTerrainFromCache();
                    SceneView.RepaintAll();
                }
            }

            if (GUILayout.Button("Open Depth Pro 2 Terrain Studio"))
            {
                DepthTerrainWindow.ShowWindow();
            }

            if (ctrl.CurrentMetadata != null)
            {
                var m = ctrl.CurrentMetadata;
                EditorGUILayout.HelpBox(
                    $"Source: {m.sourceLabel}\nModel: {m.modelName} ({m.device}, {m.precision})\nInference: {m.inferenceMs:F1} ms | Resolution: {m.rawResolution}x{m.rawResolution}",
                    MessageType.None
                );
            }
        }
    }
}
