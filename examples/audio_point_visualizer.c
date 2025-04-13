/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 * Based on examples/point.c
 *
 * Combines point cloud view with depth-based coloring and mouse navigation,
 * and modulates point depth based on system audio RMS volume using PulseAudio.
 */

#include "libfreenect.h"
#include "libfreenect_sync.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h> // For threading
#include <pulse/simple.h> // For PulseAudio
#include <pulse/error.h>  // For PulseAudio error handling

#if defined(__APPLE__)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// Define M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Configuration ---
#define AUDIO_BUFSIZE 1024 // Audio buffer size in samples
#define AUDIO_CHANNELS 2    // Number of audio channels (stereo)
#define AUDIO_RATE 44100   // Sample rate (Hz)
// float g_depth_modulation_scale_factor = 0.5f; // OLD: How much volume affects depth (Adjustable: '=' increases, '-' decreases) - REMOVED for global scaling

// --- Global Variables ---
int window;
int mx = -1, my = -1;         // Previous mouse coordinates
GLfloat anglex = 0.0f, angley = 0.0f; // Rotation angles
GLfloat zoom = 1.0f;          // Zoom factor
GLfloat min_depth_color = 650.0f; // Minimum depth for color mapping
GLfloat max_depth_color = 1050.0f; // Maximum depth for color mapping

// Rotation state
int is_continuous_rotating = 0; // 0 = inactive, 1 = active (toggled by 'r')
GLfloat g_rotation_speed = 1.0f; // Degrees per frame for continuous rotation
float g_current_rotation_speed = 0.1f; // Smoothed rotation speed
// Manual rotation override
int is_manual_rotation_override = 0; // 0 = use audio-reactive speed, 1 = use manual speed
float g_manual_rotation_speed = 0.5f; // Manual speed value
// Global Scaling State (NEW)
float g_target_scale_factor = 1.0f; // Target scale based on audio
float g_current_scale_factor = 1.0f; // Smoothed current scale applied
float g_scale_sensitivity = 25.0f; // How much RMS affects target scale (TUNE THIS)
float g_scale_smoothing_factor = 0.05f; // How fast scale adapts (lower is smoother)
const float MIN_SCALE = 1.0f;
const float MAX_SCALE = 5.0f; // Target max expansion

void animate_rotation(void); // Forward declaration for animation function

// Audio Capture State
pthread_t audio_thread;
double previous_buffer_energy = 0.0; // Energy of the previous buffer (audio thread only)
pthread_mutex_t rms_mutex;
volatile double g_current_rms_volume = 0.0; // Shared RMS volume (used for scaling)
volatile double g_onset_strength = 0.0; // Shared onset strength (for rotation)
volatile int keep_running = 1; // Flag to signal audio thread to stop
pa_simple *pa_s = NULL; // PulseAudio simple connection (managed by audio thread)
// --- Function Prototypes ---
void DrawGLScene();
void cleanup_resources(void);
void *audio_capture_thread(void *arg);

// --- OpenGL/GLUT Functions ---

// implemented here to avoid dependency on libGLU
void gluPerspective(GLdouble fovY, GLdouble aspect, GLdouble zNear, GLdouble zFar)
{
    const GLdouble fH = tan(fovY / 360 * M_PI) * zNear;
    const GLdouble fW = fH * aspect;
    glFrustum(-fW, fW, -fH, fH, zNear, zFar);
}

// Do the projection from u,v,depth to X,Y,Z directly in an opengl matrix
// These numbers come from a combination of the ros kinect_node wiki, and
// nicolas burrus' posts.
void LoadVertexMatrix()
{
    float fx = 594.21f;
    float fy = 591.04f;
    float a = -0.0030711f;
    float b = 3.3309495f;
    float cx = 339.5f;
    float cy = 242.7f;
    GLfloat mat[16] = {
        1/fx,     0,  0, 0,
        0,    -1/fy,  0, 0,
        0,       0,  0, a,
        -cx/fx, cy/fy, -1, b
    };
    glMultMatrixf(mat);
}

// Simple depth-to-color mapping (Blue=near, Red=far)
void depth_to_color(float depth, float *r, float *g, float *b) {
    float range = max_depth_color - min_depth_color;
    float normalized_depth = 0.0f;
    if (range > 0.0f) {
        normalized_depth = fminf(1.0f, fmaxf(0.0f, (depth - min_depth_color) / range));
    }
    *r = normalized_depth;         // Red increases with distance
    *g = 0.0f;
    *b = 1.0f - normalized_depth; // Blue decreases with distance
}


void no_kinect_quit(void)
{
    fprintf(stderr, "Error: Kinect not connected?\n");
    // Use the cleanup function to ensure resources are released
    exit(1); // atexit handler (cleanup_resources) should be called
}

void DrawGLScene()
{
    short *depth = 0;
    uint32_t ts;
    if (freenect_sync_get_depth((void**)&depth, &ts, 0, FREENECT_DEPTH_11BIT) < 0)
        no_kinect_quit();

    static short xyz[480][640][3];
    int i, j;
    for (i = 0; i < 480; i++) {
        for (j = 0; j < 640; j++) {
            xyz[i][j][0] = j;
            xyz[i][j][1] = i;
            xyz[i][j][2] = depth[i*640+j];
        }
    }

    // Get current RMS and Onset strength safely
    double local_rms_volume;
    double local_onset_strength; // Read onset even if not used directly here yet
    pthread_mutex_lock(&rms_mutex);
    local_rms_volume = g_current_rms_volume;
    local_onset_strength = g_onset_strength; // Keep reading onset for rotation logic
    pthread_mutex_unlock(&rms_mutex);

    // --- Centroid and Global Scale Calculation ---
    float fx = 594.21f;
    float fy = 591.04f;
    float a = -0.0030711f;
    float b = 3.3309495f;
    float cx = 339.5f;
    float cy = 242.7f;

    double sumX = 0.0, sumY = 0.0, sumZ = 0.0;
    int count = 0;
    for (i = 0; i < 480; i++) {
        for (j = 0; j < 640; j++) {
            short d_raw = depth[i*640+j];
            if (d_raw != 0 && d_raw < 2047) {
                float d = (float)d_raw;
                float u = (float)j;
                float v = (float)i;
                float W = a * d + b;
                // Avoid division by zero or near-zero (handles potential invalid projection)
                if (fabsf(W) > 1e-6f) {
                     sumX += (u / fx - cx / fx) / W;
                     sumY += (-v / fy + cy / fy) / W;
                     sumZ += -1.0f / W;
                     count++;
                }
            }
        }
    }

    float centroidX = 0.0f, centroidY = 0.0f, centroidZ = 0.0f;
    if (count > 0) {
        centroidX = (float)(sumX / count);
        centroidY = (float)(sumY / count);
        centroidZ = (float)(sumZ / count);
    }

    // --- Calculate Target and Smoothed Global Scale (NEW) ---
    // Calculate target scale based on RMS volume
    g_target_scale_factor = 1.0f + (float)local_rms_volume * g_scale_sensitivity;
    // Clamp target scale
    if (g_target_scale_factor < MIN_SCALE) g_target_scale_factor = MIN_SCALE;
    if (g_target_scale_factor > MAX_SCALE) g_target_scale_factor = MAX_SCALE;

    // Smoothly update the current scale factor towards the target
    g_current_scale_factor += g_scale_smoothing_factor * (g_target_scale_factor - g_current_scale_factor);
    // --- End Global Scale Calculation ---

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // Apply mouse transformations
    glTranslatef(0.0f, 0.0f, -3.0f * zoom); // Apply zoom centered around origin
    glRotatef(angley, 1.0f, 0.0f, 0.0f);    // Rotate based on mouse Y drag
    glRotatef(anglex, 0.0f, 1.0f, 0.0f);    // Rotate based on mouse X drag
    glTranslatef(0.0f, 0.0f, 1.5f);         // Move center of rotation slightly forward

    // Apply Kinect projection matrix
    LoadVertexMatrix();

    // Apply global scaling transformation around the centroid
    // This happens *after* LoadVertexMatrix so scaling is in world space
    glTranslatef(centroidX, centroidY, centroidZ);
    glScalef(g_current_scale_factor, g_current_scale_factor, g_current_scale_factor); // Use smoothed scale
    glTranslatef(-centroidX, -centroidY, -centroidZ);

    // Draw points
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for (i = 0; i < 480; i++) {
        for (j = 0; j < 640; j++) {
            if (xyz[i][j][2] != 0 && xyz[i][j][2] < 2047) { // Filter out invalid depth values
                float r, g, b;
                short original_d = xyz[i][j][2];
                GLshort original_u = xyz[i][j][0]; // u is j
                GLshort original_v = xyz[i][j][1]; // v is i

                depth_to_color((float)original_d, &r, &g, &b); // Color based on original depth

                // Use original depth for drawing, scaling is now handled globally
                GLshort vertex[3] = {original_u, original_v, original_d};

                glColor3f(r, g, b);
                glVertex3sv(vertex); // Use the vertex with original depth
            }
        }
    }
    glEnd();

    glutSwapBuffers();
}

void keyPressed(unsigned char key, int x, int y)
{
    float step = 10.0f; // Adjustment step for depth range

    switch (key) {
        case 27: // ESC key
            // Signal audio thread to stop and exit. Cleanup happens via atexit.
            keep_running = 0;
            // No need to explicitly call freenect_sync_stop() here if registered with atexit
            glutDestroyWindow(window);
            exit(0); // atexit handler (cleanup_resources) will be called
            break;
        case '+': // Increase max depth (red further)
            max_depth_color += step;
            printf("Depth range: [%.1f, %.1f]\n", min_depth_color, max_depth_color);
            glutPostRedisplay();
            break;
        case '-': // Decrease scale sensitivity (NEW)
            g_scale_sensitivity -= 5.0f; // Adjust step size as needed
            if (g_scale_sensitivity < 1.0f) { // Prevent zero or negative sensitivity
                g_scale_sensitivity = 1.0f;
            }
            printf("Scale Sensitivity: %.2f\n", g_scale_sensitivity);
            break;
        case ']': // Increase min depth (blue further)
             if (min_depth_color + step < max_depth_color) { // Ensure min < max
                min_depth_color += step;
                printf("Depth range: [%.1f, %.1f]\n", min_depth_color, max_depth_color);
                glutPostRedisplay();
            } else {
                 printf("Cannot increase min depth above max depth - step.\n");
            }
            break;
        case '[': // Decrease min depth (blue closer)
            min_depth_color -= step;
            if (min_depth_color < 0) min_depth_color = 0; // Prevent negative min depth
            printf("Depth range: [%.1f, %.1f]\n", min_depth_color, max_depth_color);
            glutPostRedisplay();
            break;
        case ',': // Decrease manual rotation speed and activate override
            is_manual_rotation_override = 1;
            g_manual_rotation_speed -= 0.2f;
            if (g_manual_rotation_speed < 0.0f) g_manual_rotation_speed = 0.0f; // Prevent negative speed
            printf("Manual Rotation Speed Override: %.2f\n", g_manual_rotation_speed);
            break;
        case '.': // Increase manual rotation speed and activate override
            is_manual_rotation_override = 1;
            g_manual_rotation_speed += 0.2f;
            printf("Manual Rotation Speed Override: %.2f\n", g_manual_rotation_speed);
            break;
        case 'r': // Toggle continuous rotation
        case 'R':
            is_continuous_rotating = !is_continuous_rotating; // Toggle the flag
            if (is_continuous_rotating) {
                printf("Continuous rotation: ON (Audio-Reactive Speed)\n");
                is_manual_rotation_override = 0; // Reset to audio-reactive on enable
                glutIdleFunc(&animate_rotation); // Use animation function when rotating
            } else {
                printf("Continuous rotation: OFF\n");
                glutIdleFunc(&DrawGLScene); // Use standard redraw when not rotating
            }
            break;
        case '=': // Increase scale sensitivity (NEW)
            g_scale_sensitivity += 5.0f; // Adjust step size as needed
            printf("Scale Sensitivity: %.2f\n", g_scale_sensitivity);
            break;
    }
}

// Callback for mouse button presses and scroll wheel
void mouseButtonPressed(int button, int state, int x, int y) {
    // Disable mouse interaction during continuous rotation
    if (is_continuous_rotating) {
        return;
    }
    if (state == GLUT_DOWN) {
        switch (button) {
            case GLUT_LEFT_BUTTON: // Start drag
                mx = x;
                my = y;
                break;
            case 3: // Scroll wheel up - Zoom in
                zoom /= 1.1f; // Inverted zoom compared to artistic_view for more intuitive feel
                glutPostRedisplay();
                break;
            case 4: // Scroll wheel down - Zoom out
                zoom *= 1.1f; // Inverted zoom
                glutPostRedisplay();
                break;
        }
    } else if (state == GLUT_UP && button == GLUT_LEFT_BUTTON) { // End drag
        mx = -1;
        my = -1;
    }
}

// Callback for mouse motion (drag)
void mouseMoved(int x, int y) {
    // Disable mouse rotation during continuous rotation
    if (is_continuous_rotating) {
        // Update mouse position to prevent jumps when rotation stops
        mx = x;
        my = y;
        return;
    }

    if (mx >= 0 && my >= 0) { // Only rotate if dragging
        anglex += x - mx;
        angley += y - my;
        glutPostRedisplay(); // Request redraw after mouse rotation
    }
    // Update current mouse position for next delta calculation
    mx = x;
    my = y;
}
// Idle function for handling continuous rotation
void animate_rotation() {
    if (is_continuous_rotating) {
        if (is_manual_rotation_override) {
            // Use manual speed
            anglex += g_manual_rotation_speed;
        } else {
            // Use existing smoothed audio-reactive logic
            // Read current onset strength safely
            double local_onset_strength;
            pthread_mutex_lock(&rms_mutex);
            local_onset_strength = g_onset_strength;
            pthread_mutex_unlock(&rms_mutex);

            // Calculate the target speed based on onset
            const float base_rotation_speed = 0.1f;
            const float onset_rotation_scale = 5.0f;
            float target_rotation_speed = base_rotation_speed + (local_onset_strength * onset_rotation_scale);

            // Define smoothing factor
            const float smoothing_factor = 0.05f;

            // Update the current speed using linear interpolation (lerp)
            g_current_rotation_speed = g_current_rotation_speed + smoothing_factor * (target_rotation_speed - g_current_rotation_speed);

            // Update angle using the smoothed speed
            anglex += g_current_rotation_speed;
        }
        // Optional: Keep angle within a range if desired, e.g., using fmodf
        // anglex = fmodf(anglex, 360.0f);

        glutPostRedisplay(); // Request redraw for this rotation step
    }
    // No 'else' needed, as the idle function is switched back in keyPressed when rotation stops.
}



void ReSizeGLScene(int Width, int Height)
{
    glViewport(0, 0, Width, Height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60, (GLfloat)Width / (GLfloat)Height, 0.1, 100.0); // Adjusted near/far planes
    glMatrixMode(GL_MODELVIEW);
}

void InitGL(int Width, int Height)
{
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    ReSizeGLScene(Width, Height);
}

// --- Audio Capture Thread ---
void *audio_capture_thread(void *arg) {
    (void)arg; // Unused argument

    const pa_sample_spec ss = {
        .format = PA_SAMPLE_S16LE,
        .rate = AUDIO_RATE,
        .channels = AUDIO_CHANNELS
    };
    int error;
    const char *source = "alsa_output.platform-sound.analog-stereo.monitor"; // Explicitly use monitor source

    // Create a new PulseAudio simple connection
    if (!(pa_s = pa_simple_new(NULL, "AudioPointVisualizer", PA_STREAM_RECORD, source, "System Audio Capture", &ss, NULL, NULL, &error))) {
        fprintf(stderr, "pa_simple_new() failed: %s\n", pa_strerror(error));
        goto finish;
    }

    printf("PulseAudio connection established. Capturing audio...\n");

    int16_t buf[AUDIO_BUFSIZE * AUDIO_CHANNELS];

    while (keep_running) {
        // Read audio data
        if (pa_simple_read(pa_s, buf, sizeof(buf), &error) < 0) {
            fprintf(stderr, "pa_simple_read() failed: %s\n", pa_strerror(error));
            goto finish;
        }

        // Calculate RMS volume and Energy
        double current_buffer_energy = 0.0; // Renamed from sum_sq
        int total_samples = AUDIO_BUFSIZE * AUDIO_CHANNELS;
        for (int k = 0; k < total_samples; ++k) {
            // Normalize the int16_t sample to [-1.0, 1.0] (approx)
            double normalized_sample = (double)buf[k] / 32768.0;
            current_buffer_energy += normalized_sample * normalized_sample;
        }
        // Calculate RMS
        double rms = sqrt(current_buffer_energy / total_samples);

        // Calculate Onset Strength
        double onset_strength = 0.0;
        // Avoid division by zero and handle initial state
        if (previous_buffer_energy > 1e-6) {
            onset_strength = current_buffer_energy / previous_buffer_energy;
        }
        // Clamp the strength to prevent extreme values (adjust max as needed)
        const double max_onset_strength = 10.0;
        if (onset_strength > max_onset_strength) {
            onset_strength = max_onset_strength;
        }
        // Optional: Apply a threshold - only register significant onsets
        const double onset_threshold = 1.5; // Energy must increase by 50%
        if (onset_strength < onset_threshold) {
             onset_strength = 0.0; // Ignore small increases
        } else {
             // Optional: Scale down the strength after thresholding
             onset_strength = (onset_strength - onset_threshold) / (max_onset_strength - onset_threshold); // Normalize roughly to 0-1
        }

        // Update previous energy for the next iteration
        previous_buffer_energy = current_buffer_energy;

        // Debug print removed.

        // Update shared variables safely
        pthread_mutex_lock(&rms_mutex);
        g_current_rms_volume = rms; // Keep updating RMS for depth modulation
        g_onset_strength = onset_strength; // Update onset strength
        pthread_mutex_unlock(&rms_mutex);

        // Optional: Add a small sleep if CPU usage is too high,
        // but pa_simple_read should block appropriately.
        // usleep(1000); // 1ms sleep
    }

finish:
    if (pa_s) {
        pa_simple_free(pa_s);
        pa_s = NULL; // Ensure it's marked as freed
    }
    printf("Audio capture thread finished.\n");
    return NULL;
}

// --- Cleanup Function ---
void cleanup_resources(void) {
    printf("Cleaning up resources...\n");

    // Signal the audio thread to stop
    keep_running = 0;

    // Wait for the audio thread to finish
    if (pthread_join(audio_thread, NULL)) {
        perror("Error joining audio thread");
    } else {
        printf("Audio thread joined successfully.\n");
    }

    // Destroy the mutex
    pthread_mutex_destroy(&rms_mutex);
    printf("Mutex destroyed.\n");

    // Stop freenect sync operations (important!)
    freenect_sync_stop();
    printf("Freenect sync stopped.\n");

    // Note: pa_simple resources are freed within the audio thread itself
}


// --- Main Function ---
int main(int argc, char **argv)
{
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);
    glutInitWindowSize(640, 480);
    glutInitWindowPosition(0, 0);

    window = glutCreateWindow("LibFreenect Audio Point Visualizer");

    glutDisplayFunc(&DrawGLScene);
    glutIdleFunc(&DrawGLScene); // Keep redrawing
    glutReshapeFunc(&ReSizeGLScene);
    glutKeyboardFunc(&keyPressed);
    glutMouseFunc(&mouseButtonPressed); // Register mouse button callback
    glutMotionFunc(&mouseMoved);      // Register mouse motion callback

    InitGL(640, 480);

    // Check for Kinect device
    freenect_context *f_ctx;
    if (freenect_init(&f_ctx, NULL) < 0) {
        printf("freenect_init() failed\n");
        return 1;
    }
    if (freenect_num_devices(f_ctx) == 0) {
        printf("No Kinect devices found.\n");
        freenect_shutdown(f_ctx);
        return 1;
    }
    freenect_shutdown(f_ctx); // Close context, sync functions handle device internally

    // Initialize mutex
    if (pthread_mutex_init(&rms_mutex, NULL) != 0) {
        perror("Mutex initialization failed");
        return 1;
    }

    // Register cleanup function to be called on exit
    atexit(cleanup_resources);

    // Start audio capture thread
    keep_running = 1;
    if (pthread_create(&audio_thread, NULL, audio_capture_thread, NULL)) {
        perror("Error creating audio thread");
        pthread_mutex_destroy(&rms_mutex); // Clean up mutex if thread creation failed
        return 1;
    }

    printf("Starting GLUT main loop...\n");
    glutMainLoop();

    // Cleanup is handled by the atexit handler, so no explicit calls needed here normally.
    // However, glutMainLoop might not return in some GLUT implementations.
    // The atexit handler ensures cleanup on normal exit or via exit().

    return 0; // Should not be reached if glutMainLoop runs forever
}