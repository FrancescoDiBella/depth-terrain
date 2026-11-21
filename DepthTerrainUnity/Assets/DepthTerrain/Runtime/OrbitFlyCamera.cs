using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace DepthTerrain
{
    /// <summary>
    /// Smooth orbit, pan, zoom, and WASD fly camera for inspecting reconstructed 3D Depth Pro 2 terrains.
    /// Compatible with Unity 6 New Input System and Legacy Input Manager.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public class OrbitFlyCamera : MonoBehaviour
    {
        [Header("Target & Orbit")]
        public Vector3 focusPoint = new Vector3(0f, 35f, 0f);
        public float distance = 520f;
        public float yawDegrees = 35f;
        public float pitchDegrees = 36f;

        [Header("Limits & Speed")]
        public float minDistance = 60f;
        public float maxDistance = 2400f;
        public float orbitSensitivity = 0.28f;
        public float zoomSensitivity = 0.14f;
        public float panSpeed = 0.45f;
        public float flySpeed = 240f;
        public float damping = 10f;

        /// <summary>
        /// Set true by RuntimeTerrainHUD when the pointer hovers over UI panels so dragging sliders never orbits the camera.
        /// </summary>
        public bool PointerOverUi { get; set; }

        private float _targetDistance;
        private float _targetYaw;
        private float _targetPitch;
        private Vector3 _targetFocus;
        private bool _initialized;

        private void Awake()
        {
            SyncTargetsFromFields();
            ApplyTransformImmediate();
        }

        public void SyncTargetsFromFields()
        {
            _targetDistance = distance;
            _targetYaw = yawDegrees;
            _targetPitch = pitchDegrees;
            _targetFocus = focusPoint;
            _initialized = true;
        }

        /// <summary>
        /// Frame a terrain footprint smoothly from an elevated isometric angle.
        /// </summary>
        public void FrameTerrain(float terrainSize, float terrainHeight, float yaw = 38f, float pitch = 35f, bool immediate = false)
        {
            _targetFocus = new Vector3(0f, terrainHeight * 0.28f, 0f);
            _targetDistance = Mathf.Clamp(terrainSize * 1.08f, minDistance, maxDistance);
            _targetYaw = yaw;
            _targetPitch = pitch;
            _initialized = true;

            if (immediate || !Application.isPlaying)
            {
                focusPoint = _targetFocus;
                distance = _targetDistance;
                yawDegrees = _targetYaw;
                pitchDegrees = _targetPitch;
                ApplyTransformImmediate();
            }
        }

        public void ApplyTransformImmediate()
        {
            Quaternion rot = Quaternion.Euler(pitchDegrees, yawDegrees, 0f);
            Vector3 offset = rot * new Vector3(0f, 0f, -distance);
            transform.position = focusPoint + offset;
            transform.LookAt(focusPoint, Vector3.up);
        }

        private void LateUpdate()
        {
            if (!Application.isPlaying) return;
            if (!_initialized) SyncTargetsFromFields();

            HandleInput();

            float t = 1f - Mathf.Exp(-damping * Time.unscaledDeltaTime);
            distance = Mathf.Lerp(distance, _targetDistance, t);
            yawDegrees = Mathf.Lerp(yawDegrees, _targetYaw, t);
            pitchDegrees = Mathf.Lerp(pitchDegrees, _targetPitch, t);
            focusPoint = Vector3.Lerp(focusPoint, _targetFocus, t);

            ApplyTransformImmediate();
        }

        private void HandleInput()
        {
            float dt = Time.unscaledDeltaTime;

#if ENABLE_INPUT_SYSTEM
            var mouse = Mouse.current;
            var keyboard = Keyboard.current;

            if (mouse != null && !PointerOverUi)
            {
                Vector2 delta = mouse.delta.ReadValue();
                if (mouse.leftButton.isPressed || mouse.rightButton.isPressed)
                {
                    _targetYaw += delta.x * orbitSensitivity;
                    _targetPitch = Mathf.Clamp(_targetPitch - delta.y * orbitSensitivity, 8f, 86f);
                }
                else if (mouse.middleButton.isPressed)
                {
                    PanByScreenDelta(delta);
                }

                float scrollY = mouse.scroll.ReadValue().y;
                if (Mathf.Abs(scrollY) > 0.01f)
                {
                    float factor = 1f - Mathf.Sign(scrollY) * zoomSensitivity;
                    _targetDistance = Mathf.Clamp(_targetDistance * factor, minDistance, maxDistance);
                }
            }

            if (keyboard != null)
            {
                float moveX = (keyboard.dKey.isPressed || keyboard.rightArrowKey.isPressed ? 1f : 0f)
                            - (keyboard.aKey.isPressed || keyboard.leftArrowKey.isPressed ? 1f : 0f);
                float moveZ = (keyboard.wKey.isPressed || keyboard.upArrowKey.isPressed ? 1f : 0f)
                            - (keyboard.sKey.isPressed || keyboard.downArrowKey.isPressed ? 1f : 0f);
                float moveY = (keyboard.eKey.isPressed ? 1f : 0f) - (keyboard.qKey.isPressed ? 1f : 0f);
                float boost = keyboard.leftShiftKey.isPressed ? 2.2f : 1f;

                MoveFocus(moveX, moveY, moveZ, dt * boost);

                if (keyboard.rKey.wasPressedThisFrame)
                {
                    FrameTerrain(500f, 120f);
                }
            }
#else
            if (!PointerOverUi)
            {
                if (Input.GetMouseButton(0) || Input.GetMouseButton(1))
                {
                    _targetYaw += Input.GetAxis("Mouse X") * orbitSensitivity * 10f;
                    _targetPitch = Mathf.Clamp(_targetPitch - Input.GetAxis("Mouse Y") * orbitSensitivity * 10f, 8f, 86f);
                }
                float scroll = Input.GetAxis("Mouse ScrollWheel");
                if (Mathf.Abs(scroll) > 0.001f)
                {
                    _targetDistance = Mathf.Clamp(_targetDistance * (1f - scroll * 1.5f), minDistance, maxDistance);
                }
            }

            float mx = Input.GetAxis("Horizontal");
            float mz = Input.GetAxis("Vertical");
            float my = (Input.GetKey(KeyCode.E) ? 1f : 0f) - (Input.GetKey(KeyCode.Q) ? 1f : 0f);
            MoveFocus(mx, my, mz, dt);
#endif
        }

        private void PanByScreenDelta(Vector2 delta)
        {
            Quaternion yawRot = Quaternion.Euler(0f, _targetYaw, 0f);
            Vector3 right = yawRot * Vector3.right;
            Vector3 forward = yawRot * Vector3.forward;
            float scale = panSpeed * (_targetDistance / 500f);
            _targetFocus -= right * (delta.x * scale);
            _targetFocus -= forward * (delta.y * scale);
        }

        private void MoveFocus(float x, float y, float z, float scaledDt)
        {
            if (Mathf.Abs(x) < 1e-4f && Mathf.Abs(y) < 1e-4f && Mathf.Abs(z) < 1e-4f) return;
            Quaternion yawRot = Quaternion.Euler(0f, _targetYaw, 0f);
            Vector3 dir = (yawRot * Vector3.right) * x + (yawRot * Vector3.forward) * z + Vector3.up * y;
            _targetFocus += dir * (flySpeed * scaledDt);
        }
    }
}
