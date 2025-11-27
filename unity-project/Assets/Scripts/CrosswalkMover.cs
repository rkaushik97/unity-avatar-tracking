using UnityEngine;

public class CrosswalkMover : MonoBehaviour
{
    [Header("Movement")]
    [SerializeField] private float walkSpeed = 1.0f;
    [SerializeField] private float maxDistance = 2.78f;
    
    [Header("References")]
    [SerializeField] private Animator animator;
    [SerializeField] private Transform visualPivot;

    [Header("Visual Rotation")]
    [SerializeField] private float rotationSpeed = 360f;
    
    // Runtime variables
    private Vector3 startPosition;
    private Vector3 endPosition;
    private Vector3 pathDirection;
    private bool hasReachedEndpoint = false;
    private bool isRotating = false;
    private float targetYRotation = 0f;
    
    private void Start()
    {
        // Store initial position and direction
        startPosition = transform.position;
        pathDirection = transform.forward.normalized;
        endPosition = startPosition + (pathDirection * maxDistance);
        
        // Get animator if not assigned
        if (animator == null)
            animator = GetComponent<Animator>();
            
        // If visualPivot not assigned, create it
        if (visualPivot == null)
        {
            // Create a new pivot as a child of this object
            GameObject pivotObj = new GameObject("VisualPivot");
            pivotObj.transform.SetParent(transform);
            pivotObj.transform.localPosition = Vector3.zero;
            pivotObj.transform.localRotation = Quaternion.identity;
            
            // Move all children to this pivot
            while (transform.childCount > 1)
            {
                Transform child = transform.GetChild(1); // Skip the pivot which is now child 0
                child.SetParent(pivotObj.transform);
            }
            
            visualPivot = pivotObj.transform;
        }
        
        // Initialize the animator
        animator.SetBool("isWalking", true);
        animator.SetBool("isMovingForward", true);
        animator.SetBool("isTurning", false);
        
        // Store initial rotation
        targetYRotation = visualPivot.eulerAngles.y;
    }
    
    private void Update()
    {
        // Handle smooth rotation if needed
        if (isRotating)
        {
            HandleRotation();
            return; // Don't move while rotating
        }
        
        // Don't move while turning
        if (animator.GetBool("isTurning"))
            return;
            
        // Get current direction from animator
        bool isMovingForward = animator.GetBool("isMovingForward");
        
        // Calculate movement - ROOT moves along path regardless of visual rotation
        Vector3 moveDirection = pathDirection * (isMovingForward ? 1f : -1f);
        transform.position += moveDirection * walkSpeed * Time.deltaTime;
        
        // Calculate distance along path
        float distanceFromStart = Vector3.Dot(transform.position - startPosition, pathDirection);
        
        // Update distance parameter
        float distancePercent = Mathf.Clamp01(distanceFromStart / maxDistance);
        animator.SetFloat("distancePercent", distancePercent);
        
        // Check for endpoints
        if (isMovingForward)
        {
            // Moving toward end point
            if (distanceFromStart >= maxDistance * 0.95f && !hasReachedEndpoint)
            {
                hasReachedEndpoint = true;
                TriggerTurn();
            }
        }
        else
        {
            // Moving toward start point
            if (distanceFromStart <= 0.1f && !hasReachedEndpoint)
            {
                hasReachedEndpoint = true;
                TriggerTurn();
            }
        }
        
        // Reset endpoint flag when we've moved away
        if ((isMovingForward && distanceFromStart < maxDistance * 0.8f) ||
            (!isMovingForward && distanceFromStart > 0.5f))
        {
            hasReachedEndpoint = false;
        }
    }
    
    // Handle smooth visual rotation
    private void HandleRotation()
    {
        float step = rotationSpeed * Time.deltaTime;
        
        // Get current rotation
        Vector3 currentEuler = visualPivot.eulerAngles;
        
        // Calculate shortest distance to target (handling 0-360 wrapping)
        float currentAngle = currentEuler.y;
        float delta = Mathf.DeltaAngle(currentAngle, targetYRotation);
        
        // Check if we're close enough to finish
        if (Mathf.Abs(delta) < 1.0f)
        {
            // Snap to exact rotation
            currentEuler.y = targetYRotation;
            visualPivot.eulerAngles = currentEuler;
            isRotating = false;
            animator.SetBool("isTurning", false);
            return;
        }
        
        // Apply rotation step in correct direction
        float newAngle = Mathf.MoveTowardsAngle(currentAngle, targetYRotation, step);
        currentEuler.y = newAngle;
        visualPivot.eulerAngles = currentEuler;
    }
    
    // Helper function to trigger turning
    private void TriggerTurn()
    {
        // Set animator to turning state
        animator.SetBool("isTurning", true);
        
        // Current direction from animator
        bool isMovingForward = animator.GetBool("isMovingForward");
        
        // Flip direction
        animator.SetBool("isMovingForward", !isMovingForward);
        
        // Calculate target rotation (add 180 degrees)
        targetYRotation = visualPivot.eulerAngles.y + 180f;
        
        // Start rotation
        isRotating = true;
    }
    
    // This is called from turnAndWalkBack when turning is complete (we'll ignore it now)
    public void TurnComplete()
    {
        // We're handling rotation ourselves now
    }
}