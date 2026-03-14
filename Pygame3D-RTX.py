import math
import random
import struct
import sys
import time

import pygame
import moderngl

# Screen / camera
WIDTH, HEIGHT = 1280, 720
FOV = math.pi / 3
HALF_FOV = FOV / 2
MAX_DIST = 64.0

# Lighting
MAX_LIGHTS = 8
BASE_AMBIENT = 0.05
DEBUG_SQUARE_SIZE = 10.0

# Player movement tuning
MOVE_SPEED = 2.8  # units per second
ROT_SPEED = 1.9   # radians per second

# Particle billboard shaders (needed in _create_gl)
PART_VERTEX = r"""
#version 330
in vec3 in_pos;
in float in_size;
in vec3 in_color;
in float in_alpha;

uniform vec2 resolution;
uniform vec2 playerPos;
uniform vec2 playerDir;
uniform float camOffset;
uniform float fov;

out vec3 vColor;
out float vAlpha;

void main() {
    vec2 rel = in_pos.xy - playerPos;
    float forward = dot(rel, playerDir);
    if (forward <= 0.05) {
        gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
        vColor = vec3(0.0);
        vAlpha = 0.0;
        return;
    }
    vec2 planeN = normalize(vec2(-playerDir.y, playerDir.x));
    float side = dot(rel, planeN);
    float camPlaneScale = tan(fov * 0.5);
    float ndcX = (side / forward) / camPlaneScale;
    float screenX = (ndcX * 0.5 + 0.5) * resolution.x;
    float projHeight = resolution.y / forward;
    float centerY = resolution.y * (0.5 + camOffset);
    float screenY = centerY - (in_pos.z + 0.3) * projHeight;

    float ndcY = (screenY / resolution.y) * 2.0 - 1.0;
    float ndcX2 = (screenX / resolution.x) * 2.0 - 1.0;
    gl_Position = vec4(ndcX2, ndcY, 0.0, 1.0);
    gl_PointSize = clamp(in_size * projHeight * 0.35, 2.0, 64.0);
    vColor = in_color;
    vAlpha = in_alpha;
}
"""

PART_FRAG = r"""
#version 330
in vec3 vColor;
in float vAlpha;
out vec4 fragColor;
void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;
    float falloff = smoothstep(1.0, 0.0, r2);
    fragColor = vec4(vColor * falloff, vAlpha * falloff);
}
"""

# Particle billboard shaders (defined early to use in _create_gl)

class Player:
    def __init__(self, x: float, y: float, angle: float):
        self.pos = [x, y]
        self.angle = angle

    def move(self, amount: float, world_map):
        dx = math.cos(self.angle) * amount
        dy = math.sin(self.angle) * amount
        nx = self.pos[0] + dx
        ny = self.pos[1] + dy
        if 0 <= int(ny) < len(world_map) and 0 <= int(nx) < len(world_map[0]):
            if world_map[int(ny)][int(nx)] == 0:
                self.pos[0] = nx
                self.pos[1] = ny
                return False, (nx, ny), (0.0, 0.0)
        return True, (nx, ny), (-dx, -dy)

    def rotate(self, amount: float):
        self.angle = (self.angle + amount + math.tau) % math.tau


class Lighting:
    def __init__(self, lights):
        self.lights = lights[:MAX_LIGHTS]


class Game:
    def __init__(self):
        pygame.init()
        self.fullscreen = True
        self.vsync = True
        self.clock = pygame.time.Clock()
        self.start_time = time.time()
        self.last_dt = 0.0
        self.width = WIDTH
        self.height = HEIGHT

        self.map_rows = [
            "##################",
            "#................#",
            "#.##.##.##.##...##",
            "#................#",
            "#.####.##.####...#",
            "#................#",
            "#.##.##.##.##...##",
            "#................#",
            "#.####.##.####...#",
            "#................#",
            "#.##.##.##.##...##",
            "#................#",
            "#.####.##.####...#",
            "#................#",
            "#.##.##.##.##...##",
            "#................#",
            "#.####.##.####...#",
            "##################",
        ]
        self.world_map = [[1 if ch == "#" else 0 for ch in row] for row in self.map_rows]
        self.map_size = (len(self.world_map[0]), len(self.world_map))
        self.map_dirty_every_frame = True

        self.player = Player(9.0, 9.0, 0.0)
        self.rot_sens = 1.0
        self.mouse_sens = 0.0025
        self.pitch = 0.0
        self.jump_height = 0.0
        self.jump_vel = 0.0
        self.jump_impulse = 4.2
        self.gravity = 9.8
        self.particles = []
        self.max_particles = 512
        self.lighting = Lighting([
            {"pos": [4.5, 4.5], "origin": (4.5, 4.5), "orbit": 1.4, "speed": 1.4, "phase": 0.2, "color": (255, 255, 32), "radius": 5.0, "intensity": 3.0},
            {"pos": [13.5, 4.5], "origin": (13.5, 4.5), "orbit": 1.1, "speed": 1.1, "phase": 1.1, "color": (0, 255, 255), "radius": 4.6, "intensity": 2.6},
            {"pos": [4.5, 13.5], "origin": (4.5, 13.5), "orbit": 1.7, "speed": 0.9, "phase": 2.2, "color": (255, 80, 255), "radius": 4.8, "intensity": 2.7},
            {"pos": [13.5, 13.5], "origin": (13.5, 13.5), "orbit": 1.3, "speed": 1.7, "phase": 3.4, "color": (255, 255, 255), "radius": 5.5, "intensity": 3.2},
        ])
        self.lights_paused = False
        self.debug_mode = False
        self.proc_textures_enabled = False
        self._create_gl()

    def _map_bytes(self):
        w, h = self.map_size
        data = bytearray()
        for y in range(h):
            for x in range(w):
                data.append(255 if self.world_map[y][x] else 0)
        return data

    def _spawn_particles(self, pos, normal):
        if len(self.particles) > self.max_particles:
            return
        px, py = pos
        nx, ny = normal
        nlen = math.hypot(nx, ny)
        if nlen > 0:
            nx /= nlen
            ny /= nlen
        base_dir = math.atan2(ny, nx) if nlen > 0 else self.player.angle + math.pi
        for _ in range(24):
            ang = base_dir + (random.random() - 0.5) * 1.2
            speed = random.uniform(2.0, 5.5)
            vx = math.cos(ang) * speed
            vy = math.sin(ang) * speed
            vz = random.uniform(3.5, 6.0)
            size = random.uniform(6.0, 12.0)
            life = random.uniform(0.9, 1.6)
            col = (random.uniform(1.1, 1.4), random.uniform(0.5, 0.8), random.uniform(0.1, 0.25))
            self.particles.append({
                "pos": [px, py, 0.2],
                "vel": [vx, vy, vz],
                "life": life,
                "size": size,
                "color": col,
                "age": 0.0
            })
        if len(self.particles) > self.max_particles:
            self.particles = self.particles[-self.max_particles:]

    def _update_particles(self, dt):
        g = self.gravity
        new_list = []
        for p in self.particles:
            p["age"] += dt
            if p["age"] > p["life"]:
                continue
            p["vel"][2] -= g * dt
            for i in range(3):
                p["pos"][i] += p["vel"][i] * dt
            # floor bounce
            if p["pos"][2] < 0.0:
                p["pos"][2] = 0.0
                p["vel"][2] = -p["vel"][2] * 0.45
                p["vel"][0] *= 0.78
                p["vel"][1] *= 0.78
                # slight rolling friction
            # wall simple damping (stay in map bounds)
            if p["pos"][0] < 0 or p["pos"][0] >= self.map_size[0] or p["pos"][1] < 0 or p["pos"][1] >= self.map_size[1]:
                continue
            new_list.append(p)
        self.particles = new_list

    def _build_map_texture(self):
        w, h = self.map_size
        tex = self.ctx.texture((w, h), 1, data=self._map_bytes())
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.repeat_x = False
        tex.repeat_y = False
        return tex

    def _build_noise_texture(self, size=64):
        rng = random.Random(time.time_ns())
        data = bytearray()
        for y in range(size):
            for x in range(size):
                n = rng.randint(0, 255)
                data.extend([n, n, n, 255])
        tex = self.ctx.texture((size, size), 4, data=data)
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.build_mipmaps()
        tex.repeat_x = True
        tex.repeat_y = True
        return tex

    def _create_gl(self):
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if self.fullscreen:
            flags |= pygame.FULLSCREEN
            info = pygame.display.Info()
            self.width, self.height = info.current_w, info.current_h
        else:
            self.width, self.height = WIDTH, HEIGHT
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        try:
            pygame.display.set_mode((self.width, self.height), flags, vsync=1 if self.vsync else 0)
        except TypeError:
            pygame.display.set_mode((self.width, self.height), flags)
        pygame.display.set_caption("GPU Raymarched 2.5D - ModernGL")
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()  # reset relative motion

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.program = self.ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=FRAG_SHADER,
        )

        self.particle_prog = self.ctx.program(
            vertex_shader=PART_VERTEX,
            fragment_shader=PART_FRAG,
        )

        self.quad = self.ctx.buffer(struct.pack(
            "8f",
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ))
        self.vao = self.ctx.simple_vertex_array(self.program, self.quad, "in_pos")
        self.particle_buf = self.ctx.buffer(reserve=1024 * 64)
        self.particle_vao = self.ctx.vertex_array(
            self.particle_prog,
            [
                (self.particle_buf, "3f 1f 3f 1f", "in_pos", "in_size", "in_color", "in_alpha"),
            ],
        )

        self.map_texture = self._build_map_texture()
        self.noise_texture = self._build_noise_texture()

        self.program["mapTex"].value = 0
        self.program["noiseTex"].value = 1
        self.program["resolution"].value = (self.width, self.height)
        self.program["mapSize"].value = self.map_size
        self.program["fov"].value = FOV
        self.program["debugMode"].value = 0
        self.program["debugSize"].value = DEBUG_SQUARE_SIZE
        self.program["useNoise"].value = 0
        self.program["camOffset"].value = 0.0
        self.particle_prog["resolution"].value = (self.width, self.height)
        self.particle_prog["playerPos"].value = (0.0, 0.0)
        self.particle_prog["playerDir"].value = (1.0, 0.0)
        self.particle_prog["camOffset"].value = 0.0
        self.particle_prog["fov"].value = FOV

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if hasattr(pygame, "WINDOWEVENT") and event.type == pygame.WINDOWEVENT:
                if getattr(event, "event", None) == getattr(pygame, "WINDOWEVENT_FOCUS_GAINED", 0):
                    pygame.event.set_grab(True)
                    pygame.mouse.set_visible(False)
                    pygame.mouse.get_rel()

    def handle_input(self, dt: float):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            sys.exit()
        # mouse look
        mx, my = pygame.mouse.get_rel()
        self.player.rotate(mx * self.mouse_sens * self.rot_sens)
        self.pitch = max(-0.35, min(0.35, self.pitch + my * self.mouse_sens * self.rot_sens))
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            speed = MOVE_SPEED * (1.6 if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL] else 1.0)
            hit, pos, normal = self.player.move(speed * dt, self.world_map)
            if hit:
                self._spawn_particles(pos, normal)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            speed = MOVE_SPEED * (1.6 if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL] else 1.0)
            hit, pos, normal = self.player.move(-speed * dt, self.world_map)
            if hit:
                self._spawn_particles(pos, normal)
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.player.rotate(-ROT_SPEED * self.rot_sens * dt)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.player.rotate(ROT_SPEED * self.rot_sens * dt)
        if keys[pygame.K_SPACE] and self.jump_height == 0.0 and self.jump_vel == 0.0:
            self.jump_vel = self.jump_impulse
        if keys[pygame.K_l]:
            self.lights_paused = True
        if keys[pygame.K_k]:
            self.lights_paused = False
        if keys[pygame.K_F3]:
            self.debug_mode = True
        if keys[pygame.K_F2]:
            self.debug_mode = False
        if keys[pygame.K_F10]:
            self.fullscreen = not self.fullscreen
            self._create_gl()
        if keys[pygame.K_v]:
            self.vsync = not self.vsync
            self._create_gl()
        if keys[pygame.K_t]:
            self.proc_textures_enabled = not self.proc_textures_enabled
            if self.proc_textures_enabled:
                self.noise_texture = self._build_noise_texture()
        if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]:
            self.rot_sens = min(3.0, self.rot_sens + 0.5 * dt * 10)
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            self.rot_sens = max(0.3, self.rot_sens - 0.5 * dt * 10)

    def _upload_lights(self):
        count = min(len(self.lighting.lights), MAX_LIGHTS)
        self.program["lightCount"].value = count

        pos_buf = []
        color_buf = []
        radius_buf = []
        intensity_buf = []

        for i in range(MAX_LIGHTS):
            if i < count:
                light = self.lighting.lights[i]
                pos_buf.extend([float(light["pos"][0]), float(light["pos"][1]), 0.25])
                color_buf.extend([c / 255.0 for c in light["color"]])
                radius_buf.append(float(light["radius"]))
                intensity_buf.append(float(light["intensity"]))
            else:
                pos_buf.extend([0.0, 0.0, 0.0])
                color_buf.extend([0.0, 0.0, 0.0])
                radius_buf.append(0.0)
                intensity_buf.append(0.0)

        self.program["lightPos"].write(struct.pack(f"{MAX_LIGHTS * 3}f", *pos_buf))
        self.program["lightColor"].write(struct.pack(f"{MAX_LIGHTS * 3}f", *color_buf))
        self.program["lightRadius"].write(struct.pack(f"{MAX_LIGHTS}f", *radius_buf))
        self.program["lightIntensity"].write(struct.pack(f"{MAX_LIGHTS}f", *intensity_buf))

    def _animate_lights(self):
        if self.lights_paused:
            return
        t = time.time() - self.start_time
        for i, light in enumerate(self.lighting.lights):
            origin = light.get("origin", light["pos"])
            orbit = light.get("orbit", 1.2)
            speed = light.get("speed", 0.8 + i * 0.1)
            phase = light.get("phase", i * 1.3)
            off_x = math.sin(t * speed + phase) * orbit
            off_y = math.cos(t * speed * 0.85 + phase * 0.7) * orbit
            light["pos"][0] = origin[0] + off_x
            light["pos"][1] = origin[1] + off_y

    def render(self):
        # jump physics
        dt = self.last_dt
        if self.jump_height > 0.0 or self.jump_vel != 0.0:
            self.jump_height += self.jump_vel * dt
            self.jump_vel -= self.gravity * dt
            if self.jump_height < 0.0:
                self.jump_height = 0.0
                self.jump_vel = 0.0

        self.ctx.clear(0.02, 0.03, 0.05, 1.0)
        # intentionally rebuild/write map texture every frame (no caching)
        self.map_texture.write(self._map_bytes())
        self.map_texture.use(location=0)
        if self.proc_textures_enabled:
            self.noise_texture.use(location=1)
        self.program["useNoise"].value = 1 if self.proc_textures_enabled else 0
        self.program["playerPos"].value = tuple(self.player.pos)
        self.program["playerDir"].value = (math.cos(self.player.angle), math.sin(self.player.angle))
        self.program["camOffset"].value = self.jump_height * 0.08 + self.pitch
        self._animate_lights()
        self._upload_lights()
        self.program["debugMode"].value = 1 if self.debug_mode else 0
        self.vao.render(moderngl.TRIANGLE_STRIP)
        # particles update/draw
        self._update_particles(dt)
        if self.particles:
            pdata = bytearray()
            for p in self.particles:
                # alpha fades with age
                life_k = max(0.0, 1.0 - p["age"] / p["life"])
                alpha = min(1.0, life_k * 0.9)
                pdata.extend(struct.pack("3f f 3f f",
                    p["pos"][0], p["pos"][1], p["pos"][2],
                    p["size"],
                    p["color"][0], p["color"][1], p["color"][2],
                    alpha))
            self.particle_buf.orphan(len(pdata))
            self.particle_buf.write(pdata)
            self.particle_prog["resolution"].value = (self.width, self.height)
            self.particle_prog["playerPos"].value = tuple(self.player.pos)
            self.particle_prog["playerDir"].value = (math.cos(self.player.angle), math.sin(self.player.angle))
            self.particle_prog["camOffset"].value = self.jump_height * 0.08 + self.pitch
            self.particle_prog["fov"].value = FOV
            self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)  # additive glow
            self.particle_vao.render(mode=moderngl.POINTS, vertices=len(self.particles))
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        pygame.display.flip()
        fps = self.clock.get_fps()
        pygame.display.set_caption(f"GPU Raymarched 2.5D | FPS: {fps:.1f}")

    def run(self):
        while True:
            dt = self.clock.tick(120) / 1000.0
            self.last_dt = dt
            self.handle_events()
            self.handle_input(dt)
            self.render()


VERT_SHADER = """
#version 330
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


FRAG_SHADER = """
#version 330
uniform sampler2D mapTex;
uniform vec2 resolution;
uniform vec2 mapSize;      // (width, height)
uniform vec2 playerPos;
uniform vec2 playerDir;
uniform float camOffset;
uniform float fov;
uniform int debugMode;
uniform float debugSize;
uniform int useNoise;
uniform sampler2D noiseTex;

const int MAX_LIGHTS = """ + str(MAX_LIGHTS) + """;
uniform int lightCount;
uniform vec3 lightPos[MAX_LIGHTS];
uniform vec3 lightColor[MAX_LIGHTS];
uniform float lightRadius[MAX_LIGHTS];
uniform float lightIntensity[MAX_LIGHTS];

// Poisson disk kernel for soft shadows (rotated per‑pixel for noise-free dithering)
const vec2 POISSON[8] = vec2[](
    vec2(-0.94201624, -0.39906216),
    vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870),
    vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845),
    vec2( 0.97484398,  0.75648379)
);

out vec4 fragColor;

float sampleMap(vec2 p) {
    if (p.x < 0.0 || p.y < 0.0 || p.x >= mapSize.x || p.y >= mapSize.y) return 1.0;
    return texture(mapTex, (p + 0.5) / mapSize).r;
}

struct Hit {
    bool hit;
    float dist;
    vec2 pos;
    vec2 normal;
};

Hit traceGrid(vec2 origin, vec2 dir) {
    Hit h;
    h.hit = false;
    h.dist = 1024.0;
    h.pos = origin;
    h.normal = vec2(0.0);

    vec2 mapPos = floor(origin);
    vec2 delta = abs(vec2(1.0 / dir.x, 1.0 / dir.y));
    vec2 stepDir = vec2(dir.x > 0.0 ? 1.0 : -1.0, dir.y > 0.0 ? 1.0 : -1.0);
    vec2 sideDist = vec2(
        (dir.x > 0.0 ? (mapPos.x + 1.0 - origin.x) : (origin.x - mapPos.x)) * delta.x,
        (dir.y > 0.0 ? (mapPos.y + 1.0 - origin.y) : (origin.y - mapPos.y)) * delta.y
    );

    int side = 0;
    for (int i = 0; i < 512; ++i) {
        if (sideDist.x < sideDist.y) {
            mapPos.x += stepDir.x;
            h.dist = sideDist.x;
            sideDist.x += delta.x;
            side = 0;
        } else {
            mapPos.y += stepDir.y;
            h.dist = sideDist.y;
            sideDist.y += delta.y;
            side = 1;
        }
        if (mapPos.x < 0.0 || mapPos.y < 0.0 || mapPos.x >= mapSize.x || mapPos.y >= mapSize.y) {
            break;
        }
        if (sampleMap(mapPos + vec2(0.5)) > 0.5 || sampleMap(mapPos) > 0.5) {
            h.hit = true;
            h.pos = origin + dir * h.dist;
            h.normal = side == 0 ? vec2(-stepDir.x, 0.0) : vec2(0.0, -stepDir.y);
            break;
        }
    }
    return h;
}

float ambientOcclusion(vec2 p) {
    float occ = 0.0;
    float r = 0.45;
    occ += sampleMap(p + vec2(r, 0.0));
    occ += sampleMap(p + vec2(-r, 0.0));
    occ += sampleMap(p + vec2(0.0, r));
    occ += sampleMap(p + vec2(0.0, -r));
    occ += sampleMap(p + vec2(r, r));
    occ += sampleMap(p + vec2(-r, r));
    occ += sampleMap(p + vec2(r, -r));
    occ += sampleMap(p + vec2(-r, -r));
    return clamp(1.0 - occ * 0.16, 0.2, 1.0);
}

// Cheap hash to decorrelate per‑pixel sampling patterns
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float shadowDDA(vec2 p, vec2 lp) {
    vec2 dir = lp - p;
    float maxDist = length(dir);
    vec2 rd = dir / max(maxDist, 0.0001);
    vec2 mapPos = floor(p);
    vec2 delta = abs(vec2(1.0 / rd.x, 1.0 / rd.y));
    vec2 stepDir = vec2(rd.x > 0.0 ? 1.0 : -1.0, rd.y > 0.0 ? 1.0 : -1.0);
    vec2 sideDist = vec2(
        (rd.x > 0.0 ? (mapPos.x + 1.0 - p.x) : (p.x - mapPos.x)) * delta.x,
        (rd.y > 0.0 ? (mapPos.y + 1.0 - p.y) : (p.y - mapPos.y)) * delta.y
    );
    float t = 0.0;
    for (int i = 0; i < 128; ++i) {
        if (sideDist.x < sideDist.y) {
            t = sideDist.x;
            sideDist.x += delta.x;
            mapPos.x += stepDir.x;
        } else {
            t = sideDist.y;
            sideDist.y += delta.y;
            mapPos.y += stepDir.y;
        }
        if (t > maxDist) break;
        if (mapPos.x < 0.0 || mapPos.y < 0.0 || mapPos.x >= mapSize.x || mapPos.y >= mapSize.y) break;
        if (sampleMap(mapPos + vec2(0.5)) > 0.5 || sampleMap(mapPos) > 0.5) return 0.0;
    }
    return 1.0;
}

float softShadow(vec2 p, vec2 lp, float radius) {
    float hard = shadowDDA(p, lp);
    // Early-out keeps fully occluded pixels cheap
    if (hard == 0.0) return 0.0;

    float pen = 0.0;
    float jitter = clamp(radius * 0.12, 0.08, 0.30);
    float rot = hash12(p) * 6.2831853;
    mat2 r = mat2(cos(rot), -sin(rot), sin(rot), cos(rot));
    for (int i = 0; i < 8; ++i) {
        vec2 offset = r * POISSON[i] * jitter;
        pen += shadowDDA(p, lp + offset);
    }
    pen *= 0.125; // divide by 8
    return mix(hard, pen, 0.65);
}

vec3 noiseSample(vec2 uv) {
    vec3 n = texture(noiseTex, uv).rgb;
    return n;
}

vec3 shadePoint(vec2 pos, vec3 normal, bool isWall) {
    vec3 color = vec3(""" + str(BASE_AMBIENT) + """);
    for (int i = 0; i < lightCount; i++) {
        vec3 lp = lightPos[i];
        vec3 surf = vec3(pos, normal.z); // normal.z holds surface plane sign
        vec3 L3 = lp - surf;
        float dist = length(L3);
        if (dist > lightRadius[i]) continue; // fully outside influence

        float invDist = 1.0 / max(dist, 0.001);
        vec3 Ldir = L3 * invDist;
        float atten = pow(max(0.0, 1.0 - dist / lightRadius[i]), 2.0);
        if (atten < 0.002) continue; // skip tiny contributions

        float diffuse = isWall ? max(dot(normalize(normal), Ldir), 0.08) : 0.9;
        float shadow = softShadow(pos + normal.xy * 0.05, lp.xy, lightRadius[i]);
        vec3 contrib = lightColor[i] * lightIntensity[i] * atten * diffuse * shadow;
        color += contrib;
    }
    color *= ambientOcclusion(pos);
    return color;
}

vec3 toneMap(vec3 c) {
    vec3 bloom = smoothstep(1.0, 3.2, c) * 0.5;
    c += bloom;
    c = vec3(1.0) - exp(-c * 1.35);
    return pow(c, vec3(1.0 / 2.2));
}

void main() {
    vec2 frag = gl_FragCoord.xy;
    float centerY = resolution.y * (0.5 + camOffset);
    float camPlaneScale = tan(fov * 0.5);
    float ndcX = (frag.x / resolution.x) * 2.0 - 1.0;
    vec2 plane = vec2(-playerDir.y, playerDir.x) * camPlaneScale;
    vec2 rayDir = normalize(playerDir + plane * ndcX);

    Hit hit = traceGrid(playerPos, rayDir);
    float projHeight = resolution.y / max(hit.dist, 0.0001);
    float wallTop = centerY - projHeight * 0.5;
    float wallBot = centerY + projHeight * 0.5;

    vec3 color;
    bool isWall = hit.hit && frag.y >= wallTop && frag.y <= wallBot;

    if (isWall) {
        vec3 base = vec3(0.24, 0.63, 0.92);
        if (useNoise == 1) {
            base *= mix(vec3(0.7), vec3(1.3), noiseSample(hit.pos * 0.35));
        }
        vec3 n3 = vec3(hit.normal, 0.0);
        vec3 lightCol = shadePoint(hit.pos, n3, true);
        color = base * lightCol;
    } else if (frag.y > wallBot) {
        float screenY = frag.y - centerY;
        float floorDist = resolution.y / (2.0 * max(screenY, 0.0001));
        vec2 floorPos = playerPos + rayDir * floorDist;
        float checker = step(0.5, fract(floorPos.x * 0.5 + floorPos.y * 0.5));
        vec3 base = mix(vec3(0.18, 0.18, 0.2), vec3(0.28, 0.28, 0.3), checker);
        if (useNoise == 1) {
            base *= mix(vec3(0.75), vec3(1.25), noiseSample(floorPos * 0.4));
        }
        vec3 lightCol = shadePoint(floorPos, vec3(0.0, 0.0, 1.0), false);
        color = base * lightCol;
    } else {
        float screenY = centerY - frag.y;
        float ceilDist = resolution.y / (2.0 * max(screenY, 0.0001));
        vec2 ceilPos = playerPos + rayDir * ceilDist;
        float checker = step(0.5, fract(ceilPos.x * 0.5 - ceilPos.y * 0.5));
        vec3 base = mix(vec3(0.08, 0.09, 0.1), vec3(0.17, 0.17, 0.18), checker);
        if (useNoise == 1) {
            base *= mix(vec3(0.8), vec3(1.2), noiseSample(ceilPos * 0.35));
        }
        vec3 lightCol = shadePoint(ceilPos, vec3(0.0, 0.0, -1.0), false);
        color = base * lightCol;
    }

    float depthVis = clamp(hit.dist / """ + str(MAX_DIST) + """, 0.0, 1.0);
    float shadowVis = 1.0;
    if (lightCount > 0) {
        shadowVis = softShadow(hit.pos, lightPos[0].xy, lightRadius[0]);
    }

    if (debugMode == 2) {
        fragColor = vec4(vec3(depthVis), 1.0);
        return;
    }
    if (debugMode == 3) {
        fragColor = vec4(vec3(shadowVis), 1.0);
        return;
    }

    fragColor = vec4(toneMap(color), 1.0);

    if (debugMode == 1) {
        for (int i = 0; i < lightCount; ++i) {
            vec2 rel = lightPos[i].xy - playerPos;
            float forward = dot(rel, playerDir);
            if (forward <= 0.05) continue;
            vec2 planeN = normalize(vec2(-playerDir.y, playerDir.x));
            float side = dot(rel, planeN);
            float ndc = side / (forward * camPlaneScale);
            float sx = (ndc * 0.5 + 0.5) * resolution.x;
            float sy = resolution.y * 0.5;
            if (abs(frag.x - sx) < debugSize && abs(frag.y - sy) < debugSize) {
                fragColor = vec4(1.0);
                return;
            }
        }
    }
}
"""


if __name__ == "__main__":
    Game().run()

# Particle billboard shaders
PART_VERTEX = """
#version 330
in vec3 in_pos;
in float in_size;
in vec3 in_color;
in float in_alpha;

uniform vec2 resolution;
uniform vec2 playerPos;
uniform vec2 playerDir;
uniform float camOffset;
uniform float fov;

out vec3 vColor;
out float vAlpha;

void main() {
    vec2 rel = in_pos.xy - playerPos;
    float forward = dot(rel, playerDir);
    if (forward <= 0.05) {
        gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
        vColor = vec3(0.0);
        vAlpha = 0.0;
        return;
    }
    vec2 planeN = normalize(vec2(-playerDir.y, playerDir.x));
    float side = dot(rel, planeN);
    float camPlaneScale = tan(fov * 0.5);
    float ndcX = (side / forward) / camPlaneScale;
    float screenX = (ndcX * 0.5 + 0.5) * resolution.x;
    float projHeight = resolution.y / forward;
    float centerY = resolution.y * (0.5 + camOffset);
    float screenY = centerY - (in_pos.z + 0.3) * projHeight;

    float ndcY = (screenY / resolution.y) * 2.0 - 1.0;
    float ndcX2 = (screenX / resolution.x) * 2.0 - 1.0;
    gl_Position = vec4(ndcX2, ndcY, 0.0, 1.0);
    gl_PointSize = clamp(in_size * projHeight * 0.35, 2.0, 64.0);
    vColor = in_color;
    vAlpha = in_alpha;
}
"""

PART_FRAG = """
#version 330
in vec3 vColor;
in float vAlpha;
out vec4 fragColor;
void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;
    float falloff = smoothstep(1.0, 0.0, r2);
    fragColor = vec4(vColor * falloff, vAlpha * falloff);
}
"""
