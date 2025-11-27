using UnityEngine;
using System.Linq;

[RequireComponent(typeof(Animator))]
public class WalkPath : MonoBehaviour
{
    [Header("Path")]
    public Transform[] points;
    public float moveSpeed = 0.1f;
    public float rotationSpeed = 0.7f;
    public float animationScale = 8f;

    [Header("Turn180 settings")]
    public string turnClipName = "Turn180";
    public bool useRootMotion = false;    // set true if clip rotates the root and you want the Transform driven by the animation
    public float turnThresholdAngle = 150f; // angle above which we consider it a 'turn 180'
    public float fallbackTurnDuration = 0.8f;

    private Animator animator;
    private int currentPoint = 0;

    // Turn state
    private bool isTurning = false;
    private float turnDuration;
    private float turnTimer;
    private Quaternion startRot;
    private Quaternion targetRot;

    void Start()
    {
        animator = GetComponent<Animator>();

        // Try to auto-read the clip length
        turnDuration = fallbackTurnDuration;
        if (animator.runtimeAnimatorController != null && !string.IsNullOrEmpty(turnClipName))
        {
            var clip = animator.runtimeAnimatorController.animationClips
                         .FirstOrDefault(c => c.name == turnClipName);
            if (clip != null)
                turnDuration = clip.length;
            else
                Debug.LogWarning($"[WalkPath] Turn clip '{turnClipName}' not found in Animator controller. Using fallback duration {turnDuration}s.");
        }

        // If you expect root motion to handle rotation, ensure Apply Root Motion is toggled on the Animator in the Inspector.
        // We do not force it here; set it in the Inspector to match useRootMotion.
    }

    void Update()
    {
        if (points == null || points.Length == 0) return;

        // If currently turning, handle turn progression and return (no movement)
        if (isTurning)
        {
            HandleTurning();
            return;
        }

        Transform target = points[currentPoint];
        Vector3 direction = target.position - transform.position;
        float distance = direction.magnitude;

        if (distance > 0.05f)
        {
            Vector3 desiredDir = direction.normalized;
            float absAngle = Mathf.Abs(Vector3.SignedAngle(transform.forward, desiredDir, Vector3.up));

            // Large angle -> trigger Turn180
            if (absAngle >= turnThresholdAngle)
            {
                StartTurn(desiredDir);
                return; // skip movement this frame; turning starts
            }

            // Normal small-angle movement
            Vector3 move = desiredDir * moveSpeed * Time.deltaTime;
            transform.position += move;

            Quaternion lookRotation = Quaternion.LookRotation(desiredDir);
            transform.rotation = Quaternion.Slerp(transform.rotation, lookRotation, rotationSpeed * Time.deltaTime);

            float animSpeed = Mathf.Clamp01(moveSpeed * animationScale);
            animator.SetFloat("Speed", animSpeed);
        }
        else
        {
            // Reached point, idle then advance
            animator.SetFloat("Speed", 0f);
            currentPoint = (currentPoint + 1) % points.Length;
        }
    }

    void StartTurn(Vector3 desiredDirection)
    {
        // Prepare start/target rotations
        startRot = transform.rotation;
        targetRot = Quaternion.LookRotation(desiredDirection.normalized, Vector3.up);

        // Option: force exact 180° rotation instead of facing desired direction:
        // targetRot = startRot * Quaternion.Euler(0f, 180f, 0f);

        // Trigger animator
        animator.SetTrigger("Turn180");

        // Enter turning state
        isTurning = true;
        turnTimer = 0f;
        // turnDuration already read in Start()
    }

    void HandleTurning()
    {
        // If using root motion, animation should rotate the transform; we simply wait for completion
        turnTimer += Time.deltaTime;
        float t = Mathf.Clamp01(turnTimer / Mathf.Max(0.0001f, turnDuration));

        if (!useRootMotion)
        {
            // Scripted rotation to match animation progress
            transform.rotation = Quaternion.Slerp(startRot, targetRot, t);
        }
        else
        {
            // If root motion is in use, let animation drive rotation.
            // We still optionally snap at the end for precision.
        }

        // Freeze locomotion param while turning
        animator.SetFloat("Speed", 0f);

        // End when clip duration reached (a robust approach even if you also wire an Animation Event)
        if (t >= 1f)
        {
            FinishTurn();
        }
    }

    // Call this from an Animation Event placed at last frame of the Turn180 clip (recommended for root motion)
    public void OnTurnFinished()
    {
        // If Animation Event is used, call FinishTurn() to end turn immediately.
        FinishTurn();
    }

    private void FinishTurn()
    {
        isTurning = false;
        turnTimer = 0f;
        // Snap to target to avoid tiny mismatch
        transform.rotation = targetRot;

        // Optionally resume moving immediately by setting animator Speed again next frame
        // animator.SetFloat("Speed", Mathf.Clamp01(moveSpeed * animationScale));
    }
}

