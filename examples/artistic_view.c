/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2010 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "libfreenect.h"

// Define M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _MSC_VER
#define HAVE_STRUCT_TIMESPEC
#endif
#include <pthread.h>

#if defined(__APPLE__)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// implemented here to avoid dependency on libGLU
void gluPerspective(GLdouble fovY, GLdouble aspect, GLdouble zNear, GLdouble zFar)
{
    const double pi = 3.14159265358979323846;
    const GLdouble fH = tan(fovY / 360 * pi) * zNear;
    const GLdouble fW = fH * aspect;
    glFrustum(-fW, fW, -fH, fH, zNear, zFar);
}

pthread_t freenect_thread;
volatile int die = 0;

int g_argc;
char **g_argv;

int window;

pthread_mutex_t gl_backbuf_mutex = PTHREAD_MUTEX_INITIALIZER;

// back: owned by libfreenect (implicit for depth)
// mid: owned by callbacks, "latest frame ready"
// front: owned by GL, "currently being drawn"
uint8_t *depth_mid, *depth_front;
uint8_t *rgb_back, *rgb_mid, *rgb_front;

GLuint gl_depth_tex;
GLuint gl_rgb_tex;
int tilt_changed = 0;

// Artistic effect parameters
int effect_mode = 0;  // 0=normal, 1=wave, 2=pulse, 3=kaleidoscope, 4=particles
float effect_time = 0.0;
float effect_intensity = 0.5;
int show_depth_only = 0;
int depth_as_points = 0;
float point_size = 2.0;

// Mouse control variables
GLfloat anglex = 0.0f, angley = 0.0f; // Rotation angles
GLfloat zoom = 1.0f;               // Zoom factor
int mx = -1, my = -1;              // Previous mouse coordinates

freenect_context *f_ctx;
freenect_device *f_dev;
int freenect_angle = 0;
int freenect_led;

freenect_video_format requested_format = FREENECT_VIDEO_RGB;
freenect_video_format current_format = FREENECT_VIDEO_RGB;

pthread_cond_t gl_frame_cond = PTHREAD_COND_INITIALIZER;
int got_rgb = 0;
int got_depth = 0;

// Apply wave effect to depth data
void apply_wave_effect(uint8_t *depth_buffer, float time) {
    int i, j;
    for (i = 0; i < 480; i++) {
        for (j = 0; j < 640; j++) {
            int idx = 3 * (i * 640 + j);
            float wave = sin(j * 0.01 + time) * sin(i * 0.01 + time) * effect_intensity * 128;
            
            depth_buffer[idx + 0] = (uint8_t)fmin(255, fmax(0, depth_buffer[idx + 0] + wave));
            depth_buffer[idx + 1] = (uint8_t)fmin(255, fmax(0, depth_buffer[idx + 1] + wave));
            depth_buffer[idx + 2] = (uint8_t)fmin(255, fmax(0, depth_buffer[idx + 2] + wave));
        }
    }
}

// Apply pulse effect to depth data
void apply_pulse_effect(uint8_t *depth_buffer, float time) {
    int i;
    float pulse = sin(time * 3.0) * effect_intensity * 128;
    
    for (i = 0; i < 640 * 480 * 3; i++) {
        depth_buffer[i] = (uint8_t)fmin(255, fmax(0, depth_buffer[i] + pulse));
    }
}

// Apply kaleidoscope effect to RGB data
void apply_kaleidoscope_effect(uint8_t *rgb_buffer, float time) {
    uint8_t temp_buffer[640 * 480 * 3];
    memcpy(temp_buffer, rgb_buffer, 640 * 480 * 3);
    
    int center_x = 320 + sin(time) * 100;
    int center_y = 240 + cos(time) * 100;
    int segments = 6 + (int)(sin(time * 0.5) * 4);
    float angle_step = 2 * 3.14159265358979323846 / segments;
    
    int i, j;
    for (i = 0; i < 480; i++) {
        for (j = 0; j < 640; j++) {
            int dx = j - center_x;
            int dy = i - center_y;
            float angle = atan2(dy, dx);
            float distance = sqrt(dx * dx + dy * dy);
            // Normalize angle to 0-2π range
            if (angle < 0) angle += 2 * 3.14159265358979323846;
            
            
            // Get segment number and normalize angle within segment
            int segment = (int)(angle / angle_step);
            float segment_angle = angle - segment * angle_step;
            
            // Reflect if in odd segments
            if (segment % 2 == 1) {
                segment_angle = angle_step - segment_angle;
            }
            
            // Calculate source coordinates
            float src_angle = segment_angle;
            int src_x = center_x + cos(src_angle) * distance;
            int src_y = center_y + sin(src_angle) * distance;
            
            // Bounds check
            if (src_x >= 0 && src_x < 640 && src_y >= 0 && src_y < 480) {
                int dst_idx = 3 * (i * 640 + j);
                int src_idx = 3 * (src_y * 640 + src_x);
                
                rgb_buffer[dst_idx + 0] = temp_buffer[src_idx + 0];
                rgb_buffer[dst_idx + 1] = temp_buffer[src_idx + 1];
                rgb_buffer[dst_idx + 2] = temp_buffer[src_idx + 2];
            }
        }
    }
}

void DrawGLScene() {
    pthread_mutex_lock(&gl_backbuf_mutex);

    // When using YUV_RGB mode, RGB frames only arrive at 15Hz, so we shouldn't force them to draw in lock-step.
    // However, this is CPU/GPU intensive when we are receiving frames in lockstep.
    if (current_format == FREENECT_VIDEO_YUV_RGB) {
        while (!got_depth && !got_rgb) {
            pthread_cond_wait(&gl_frame_cond, &gl_backbuf_mutex);
        }
    } else {
        while ((!got_depth || !got_rgb) && requested_format != current_format) {
            pthread_cond_wait(&gl_frame_cond, &gl_backbuf_mutex);
        }
    }

    if (requested_format != current_format) {
        pthread_mutex_unlock(&gl_backbuf_mutex);
        return;
    }

    uint8_t *tmp;

    if (got_depth) {
        tmp = depth_front;
        depth_front = depth_mid;
        depth_mid = tmp;
        got_depth = 0;
    }
    if (got_rgb) {
        tmp = rgb_front;
        rgb_front = rgb_mid;
        rgb_mid = tmp;
        got_rgb = 0;
    }

    pthread_mutex_unlock(&gl_backbuf_mutex);

    // Update effect time
    effect_time += 0.05;
    
    // Apply artistic effects based on mode
    if (effect_mode == 1) {
        apply_wave_effect(depth_front, effect_time);
    } else if (effect_mode == 2) {
        apply_pulse_effect(depth_front, effect_time);
    } else if (effect_mode == 3) {
        apply_kaleidoscope_effect(rgb_front, effect_time);
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    if (depth_as_points) {
        // Draw depth as 3D point cloud
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45, 640.0/480.0, 1.0, 10000.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0, 0, -1000);
        glRotatef(180, 1, 0, 0); // Flip to correct orientation
        // Apply mouse zoom and rotation
        glScalef(zoom, zoom, 1.0f);
        glRotatef(angley, 1.0f, 0.0f, 0.0f); // Rotate based on mouse Y drag
        glRotatef(anglex, 0.0f, 1.0f, 0.0f); // Rotate based on mouse X drag
        
        glPointSize(point_size);
        glBegin(GL_POINTS);
        
        int i, j;
        for (i = 0; i < 480; i += 2) {
            for (j = 0; j < 640; j += 2) {
                int idx = 3 * (i * 640 + j);
                float depth_value = (depth_front[idx] + depth_front[idx+1] + depth_front[idx+2]) / 3.0f;
                if (depth_value > 0) {
                    // Scale x and y to maintain aspect ratio and prevent stretching
                    float x = (j - 320) * 0.8;
                    float y = (i - 240) * 0.8;
                    float z = depth_value * 1.5;
                    
                    
                    // Add some artistic movement to points
                    if (effect_mode == 4) {
                        x += sin(effect_time + i * 0.01) * effect_intensity * 20;
                        y += cos(effect_time + j * 0.01) * effect_intensity * 20;
                    }
                    
                    glColor3ub(depth_front[idx], depth_front[idx+1], depth_front[idx+2]);
                    glVertex3f(x, y, -z);
                }
            }
        }
        
        glEnd();
    } else {
        // Draw depth and RGB images side by side
        if (!show_depth_only) {
            // Draw depth image
            glBindTexture(GL_TEXTURE_2D, gl_depth_tex);
            glTexImage2D(GL_TEXTURE_2D, 0, 3, 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, depth_front);
            
            glBegin(GL_TRIANGLE_FAN);
            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
            glTexCoord2f(0, 0); glVertex3f(0, 0, 0);
            glTexCoord2f(1, 0); glVertex3f(640, 0, 0);
            glTexCoord2f(1, 1); glVertex3f(640, 480, 0);
            glTexCoord2f(0, 1); glVertex3f(0, 480, 0);
            glEnd();
            
            // Draw RGB image
            glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
            if (current_format == FREENECT_VIDEO_RGB || current_format == FREENECT_VIDEO_YUV_RGB)
                glTexImage2D(GL_TEXTURE_2D, 0, 3, 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_front);
            else
                glTexImage2D(GL_TEXTURE_2D, 0, 1, 640, 480, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, rgb_front+640*4);
            
            glBegin(GL_TRIANGLE_FAN);
            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
            glTexCoord2f(0, 0); glVertex3f(640, 0, 0);
            glTexCoord2f(1, 0); glVertex3f(1280, 0, 0);
            glTexCoord2f(1, 1); glVertex3f(1280, 480, 0);
            glTexCoord2f(0, 1); glVertex3f(640, 480, 0);
            glEnd();
        } else {
            // Draw only depth image, full screen
            glBindTexture(GL_TEXTURE_2D, gl_depth_tex);
            glTexImage2D(GL_TEXTURE_2D, 0, 3, 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, depth_front);
            
            glBegin(GL_TRIANGLE_FAN);
            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
            glTexCoord2f(0, 0); glVertex3f(0, 0, 0);
            glTexCoord2f(1, 0); glVertex3f(1280, 0, 0);
            glTexCoord2f(1, 1); glVertex3f(1280, 480, 0);
            glTexCoord2f(0, 1); glVertex3f(0, 480, 0);
            glEnd();
        }
    }
    
    // Draw effect mode info
    char effect_str[256];
    sprintf(effect_str, "Effect: %s (Press 1-5 to change)", 
            effect_mode == 0 ? "Normal" : 
            effect_mode == 1 ? "Wave" : 
            effect_mode == 2 ? "Pulse" : 
            effect_mode == 3 ? "Kaleidoscope" : 
            "Particles");
    
    glColor3f(1.0, 1.0, 1.0);
    glRasterPos2f(10, 20);
    int i;
    for (i = 0; i < strlen(effect_str); i++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, effect_str[i]);
    }
    
    char controls_str[256];
    sprintf(controls_str, "Intensity: %.1f (Press +/- to change) | Press D for depth only | Press P for point cloud", effect_intensity);
    glRasterPos2f(10, 40);
    for (i = 0; i < strlen(controls_str); i++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, controls_str[i]);
    }
    
    glutSwapBuffers();
}

void keyPressed(unsigned char key, int x, int y) {
    if (key == 27) {
        die = 1;
        pthread_join(freenect_thread, NULL);
        glutDestroyWindow(window);
        free(depth_mid);
        free(depth_front);
        free(rgb_back);
        free(rgb_mid);
        free(rgb_front);
        exit(0);
    }
    if (key == '1') {
        effect_mode = 0; // Normal
    }
    if (key == '2') {
        effect_mode = 1; // Wave
    }
    if (key == '3') {
        effect_mode = 2; // Pulse
    }
    if (key == '4') {
        effect_mode = 3; // Kaleidoscope
    }
    if (key == '5') {
        effect_mode = 4; // Particles
        depth_as_points = 1;
    }
    if (key == '+' || key == '=') {
        effect_intensity = fmin(1.0, effect_intensity + 0.1);
    }
    if (key == '-' || key == '_') {
        effect_intensity = fmax(0.0, effect_intensity - 0.1);
    }
    if (key == 'd' || key == 'D') {
        show_depth_only = !show_depth_only;
    }
    if (key == 'p' || key == 'P') {
        depth_as_points = !depth_as_points;
    }
    if (key == '[') {
        point_size = fmax(1.0, point_size - 0.5);
    }
    if (key == ']') {
        point_size = fmin(10.0, point_size + 0.5);
    }
    if (key == 'w') {
        freenect_angle++;
        if (freenect_angle > 30) {
            freenect_angle = 30;
        }
        tilt_changed++;
    }
    if (key == 's') {
        freenect_angle = 0;
        tilt_changed++;
    }
    if (key == 'f') {
        if (requested_format == FREENECT_VIDEO_IR_8BIT)
            requested_format = FREENECT_VIDEO_RGB;
        else if (requested_format == FREENECT_VIDEO_RGB)
            requested_format = FREENECT_VIDEO_YUV_RGB;
        else
            requested_format = FREENECT_VIDEO_IR_8BIT;
    }
    if (key == 'x') {
        freenect_angle--;
        if (freenect_angle < -30) {
            freenect_angle = -30;
        }
        tilt_changed++;
    }
    if (tilt_changed) {
        freenect_set_tilt_degs(f_dev, freenect_angle);
        tilt_changed = 0;
    }
}

// Callback for mouse button presses and scroll wheel
void mouseButtonPressed(int button, int state, int x, int y) {
    if (state == GLUT_DOWN) {
        switch (button) {
            case GLUT_LEFT_BUTTON: // Start drag
                mx = x;
                my = y;
                break;
            case 3: // Scroll wheel up - Zoom in
                zoom *= 1.1f;
                break;
            case 4: // Scroll wheel down - Zoom out
                zoom /= 1.1f;
                if (zoom < 0.1f) zoom = 0.1f; // Prevent zooming too far out
                break;
        }
    } else if (state == GLUT_UP && button == GLUT_LEFT_BUTTON) { // End drag
        mx = -1;
        my = -1;
    }
}

// Callback for mouse motion (drag)
void mouseMoved(int x, int y) {
    if (mx >= 0 && my >= 0) { // Only rotate if dragging
        anglex += x - mx;
        angley += y - my;
    }
    // Update current mouse position
    mx = x;
    my = y;
}

void ReSizeGLScene(int Width, int Height) {
    glViewport(0, 0, Width, Height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1280, 480, 0, -1.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void InitGL(int Width, int Height) {
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_TEXTURE_2D);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glShadeModel(GL_SMOOTH);

    glGenTextures(1, &gl_depth_tex);
    glBindTexture(GL_TEXTURE_2D, gl_depth_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &gl_rgb_tex);
    glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    ReSizeGLScene(Width, Height);
}

void *gl_threadfunc(void *arg) {
    printf("GL thread\n");

    glutInit(&g_argc, g_argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);
    glutInitWindowSize(1280, 480);
    glutInitWindowPosition(0, 0);

    window = glutCreateWindow("Kinect Artistic View");

    glutDisplayFunc(&DrawGLScene);
    glutIdleFunc(&DrawGLScene);
    glutReshapeFunc(&ReSizeGLScene);
    glutKeyboardFunc(&keyPressed);
    glutMouseFunc(&mouseButtonPressed); // Register mouse button callback
    glutMotionFunc(&mouseMoved);      // Register mouse motion callback

    InitGL(1280, 480);

    glutMainLoop();

    return NULL;
}

uint16_t t_gamma[2048];

void depth_cb(freenect_device *dev, void *v_depth, uint32_t timestamp) {
    int i;
    uint16_t *depth = (uint16_t*)v_depth;

    pthread_mutex_lock(&gl_backbuf_mutex);
    for (i=0; i<640*480; i++) {
        int pval = t_gamma[depth[i]];
        int lb = pval & 0xff;
        switch (pval>>8) {
            case 0:
                depth_mid[3*i+0] = 255;
                depth_mid[3*i+1] = 255-lb;
                depth_mid[3*i+2] = 255-lb;
                break;
            case 1:
                depth_mid[3*i+0] = 255;
                depth_mid[3*i+1] = lb;
                depth_mid[3*i+2] = 0;
                break;
            case 2:
                depth_mid[3*i+0] = 255-lb;
                depth_mid[3*i+1] = 255;
                depth_mid[3*i+2] = 0;
                break;
            case 3:
                depth_mid[3*i+0] = 0;
                depth_mid[3*i+1] = 255;
                depth_mid[3*i+2] = lb;
                break;
            case 4:
                depth_mid[3*i+0] = 0;
                depth_mid[3*i+1] = 255-lb;
                depth_mid[3*i+2] = 255;
                break;
            case 5:
                depth_mid[3*i+0] = 0;
                depth_mid[3*i+1] = 0;
                depth_mid[3*i+2] = 255-lb;
                break;
            default:
                depth_mid[3*i+0] = 0;
                depth_mid[3*i+1] = 0;
                depth_mid[3*i+2] = 0;
                break;
        }
    }
    got_depth++;
    pthread_cond_signal(&gl_frame_cond);
    pthread_mutex_unlock(&gl_backbuf_mutex);
}

void rgb_cb(freenect_device *dev, void *rgb, uint32_t timestamp) {
    pthread_mutex_lock(&gl_backbuf_mutex);

    // swap buffers
    assert (rgb_back == rgb);
    rgb_back = rgb_mid;
    freenect_set_video_buffer(dev, rgb_back);
    rgb_mid = (uint8_t*)rgb;

    got_rgb++;
    pthread_cond_signal(&gl_frame_cond);
    pthread_mutex_unlock(&gl_backbuf_mutex);
}

void *freenect_threadfunc(void *arg) {
    int accelCount = 0;

    freenect_set_tilt_degs(f_dev,freenect_angle);
    freenect_set_led(f_dev,LED_RED);
    freenect_set_depth_callback(f_dev, depth_cb);
    freenect_set_video_callback(f_dev, rgb_cb);
    freenect_set_video_mode(f_dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, current_format));
    freenect_set_depth_mode(f_dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_11BIT));
    freenect_set_video_buffer(f_dev, rgb_back);

    freenect_start_depth(f_dev);
    freenect_start_video(f_dev);

    printf("Kinect Artistic View Controls:\n");
    printf("  1-5: Change effect mode\n");
    printf("  +/-: Adjust effect intensity\n");
    printf("  D: Toggle depth-only view\n");
    printf("  P: Toggle point cloud view\n");
    printf("  [/]: Adjust point size\n");
    printf("  A/D: Rotate camera (in point cloud mode)\n");
    printf("  R: Reset camera angle\n");
    printf("  W/X: Tilt Kinect up/down\n");
    printf("  ESC: Exit\n");

    while (!die && freenect_process_events(f_ctx) >= 0) {
        
        // Throttle the text output
        if (accelCount++ >= 2000) {
            accelCount = 0;
            freenect_raw_tilt_state* state;
            freenect_update_tilt_state(f_dev);
            state = freenect_get_tilt_state(f_dev);
            double dx,dy,dz;
            freenect_get_mks_accel(state, &dx, &dy, &dz);
            printf("\r raw acceleration: %4d %4d %4d  mks acceleration: %4f %4f %4f", state->accelerometer_x, state->accelerometer_y, state->accelerometer_z, dx, dy, dz);
            fflush(stdout);
        }

        if (requested_format != current_format) {
            freenect_stop_video(f_dev);
            freenect_set_video_mode(f_dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, requested_format));
            freenect_start_video(f_dev);
            current_format = requested_format;
        }
    }

    printf("\nshutting down streams...\n");

    freenect_stop_depth(f_dev);
    freenect_stop_video(f_dev);

    freenect_close_device(f_dev);
    freenect_shutdown(f_ctx);

    printf("-- done!\n");
    return NULL;
}

int main(int argc, char **argv) {
    int res;

    depth_mid = (uint8_t*)malloc(640*480*3);
    depth_front = (uint8_t*)malloc(640*480*3);
    rgb_back = (uint8_t*)malloc(640*480*3);
    rgb_mid = (uint8_t*)malloc(640*480*3);
    rgb_front = (uint8_t*)malloc(640*480*3);

    printf("Kinect Artistic View\n");

    int i;
    for (i=0; i<2048; i++) {
        float v = i/2048.0;
        v = powf(v, 3)* 6;
        t_gamma[i] = v*6*256;
    }

    g_argc = argc;
    g_argv = argv;

    if (freenect_init(&f_ctx, NULL) < 0) {
        printf("freenect_init() failed\n");
        return 1;
    }

    freenect_set_log_level(f_ctx, FREENECT_LOG_DEBUG);
    freenect_select_subdevices(f_ctx, (freenect_device_flags)(FREENECT_DEVICE_MOTOR | FREENECT_DEVICE_CAMERA));

    int nr_devices = freenect_num_devices (f_ctx);
    printf ("Number of devices found: %d\n", nr_devices);

    int user_device_number = 0;
    if (argc > 1)
        user_device_number = atoi(argv[1]);

    if (nr_devices < 1) {
        freenect_shutdown(f_ctx);
        return 1;
    }

    if (freenect_open_device(f_ctx, &f_dev, user_device_number) < 0) {
        printf("Could not open device\n");
        freenect_shutdown(f_ctx);
        return 1;
    }

    res = pthread_create(&freenect_thread, NULL, freenect_threadfunc, NULL);
    if (res) {
        printf("pthread_create failed\n");
        freenect_shutdown(f_ctx);
        return 1;
    }

    // OS X requires GLUT to run on the main thread
    gl_threadfunc(NULL);

    return 0;
}