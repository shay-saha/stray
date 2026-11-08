#include "earth_tiles.hpp"

#include <curl/curl.h>
#include <imgui_impl_opengl3_loader.h>
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <compare>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <numbers>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace wildlife {
namespace {

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

constexpr int tile_size = 256;
constexpr int grid_size = 10;
constexpr int atlas_size = tile_size * grid_size;
constexpr int min_zoom = 6;
constexpr int max_zoom = 14;
constexpr double mercator_limit = 0.99627207622075;
constexpr double tau = 2.0 * std::numbers::pi;
constexpr GLenum rgb = 0x1907;
constexpr std::size_t max_tile_bytes = 1024 * 1024;
constexpr std::uintmax_t disk_budget = 512 * 1024 * 1024;

struct TileKey {
  int level = -1;
  int x = 0;
  int y = 0;
  auto operator<=>(const TileKey &) const = default;
};

struct TileWindow {
  int level;
  int left;
  int top;
  int right;
  int bottom;
  double center_x;
  double center_y;
};

struct Tile {
  TileKey key;
  std::vector<unsigned char> photo;
  std::vector<unsigned char> elevation;
  bool elevation_request = false;
};
struct TileRequest {
  TileKey key;
  bool elevation = false;
  auto operator<=>(const TileRequest &) const = default;
};

struct Slot {
  TileKey key;
  std::uint64_t age = 0;
  unsigned char layers = 0;
  Clock::time_point arrival;
  std::vector<unsigned char> elevation;
  std::vector<unsigned char> photo;
};

double mercator_y(double y) {
  return 0.5 - std::atanh(std::clamp(y, -mercator_limit, mercator_limit)) / tau;
}

std::optional<TileWindow> choose_window(ImVec2 center, float radius, ImVec2 min,
                                        ImVec2 max, Vec3 east, Vec3 north,
                                        Vec3 forward) {
  if (radius < std::max(max.x - min.x, max.y - min.y))
    return std::nullopt;
  const auto direction = [&](double screen_x, double screen_y) -> Vec3 {
    const double x = (screen_x - center.x) / radius;
    const double y = (center.y - screen_y) / radius;
    const double z = std::sqrt(std::max(0.0, 1.0 - x * x - y * y));
    return {static_cast<float>(east.x * x + north.x * y + forward.x * z),
            static_cast<float>(north.y * y + forward.y * z),
            static_cast<float>(east.z * x + north.z * y + forward.z * z)};
  };
  const auto focus =
      direction(std::midpoint(min.x, max.x), std::midpoint(min.y, max.y));
  if (std::abs(focus.y) >= mercator_limit)
    return std::nullopt;
  const double center_x = 0.5 + std::atan2(focus.z, focus.x) / tau;
  const double center_y = mercator_y(focus.y);
  double left = center_x, right = center_x, top = center_y, bottom = center_y;
  for (int row = 0; row <= 4; ++row) {
    for (int col = 0; col <= 4; ++col) {
      const auto point = direction(std::lerp(min.x, max.x, col / 4.0f),
                                   std::lerp(min.y, max.y, row / 4.0f));
      const double u = 0.5 + std::atan2(point.z, point.x) / tau;
      const double wrapped = u - std::round(u - center_x);
      const double v = mercator_y(point.y);
      left = std::min(left, wrapped);
      right = std::max(right, wrapped);
      top = std::min(top, v);
      bottom = std::max(bottom, v);
    }
  }

  const double scale =
      std::clamp(ImGui::GetIO().DisplayFramebufferScale.x, 1.0f, 2.0f);
  const double pixel_circumference =
      tau * radius * std::hypot(focus.x, focus.z) * scale;
  const int target = std::clamp(
      static_cast<int>(std::ceil(std::log2(pixel_circumference / tile_size))),
      min_zoom, max_zoom);
  for (int level = target; level >= min_zoom; --level) {
    const int side = 1 << level;
    const int x0 = static_cast<int>(std::floor(left * side));
    const int x1 = static_cast<int>(std::floor(right * side));
    const int y0 =
        std::clamp(static_cast<int>(std::floor(top * side)), 0, side - 1);
    const int y1 =
        std::clamp(static_cast<int>(std::floor(bottom * side)), 0, side - 1);
    if (x1 - x0 < grid_size && y1 - y0 < grid_size) {
      return TileWindow{level,          x0, y0, x1, y1, center_x * side,
                        center_y * side};
    }
  }
  return std::nullopt;
}

std::vector<unsigned char> read_bytes(const fs::path &path) {
  std::error_code error;
  const auto length = fs::file_size(path, error);
  if (error || length == 0 || length > max_tile_bytes)
    return {};
  std::ifstream file(path, std::ios::binary);
  std::vector<unsigned char> bytes(static_cast<std::size_t>(length));
  if (!file.read(reinterpret_cast<char *>(bytes.data()),
                 static_cast<std::streamsize>(length)))
    return {};
  return bytes;
}

std::vector<unsigned char> decode(const std::vector<unsigned char> &bytes) {
  if (bytes.empty())
    return {};
  int width, height, channels;
  if (!stbi_info_from_memory(bytes.data(), static_cast<int>(bytes.size()),
                             &width, &height, &channels) ||
      width != tile_size || height != tile_size)
    return {};
  std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels(
      stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()),
                            &width, &height, nullptr, 3),
      stbi_image_free);
  if (!pixels)
    return {};
  return {pixels.get(), pixels.get() + tile_size * tile_size * 3};
}

std::size_t receive(char *contents, std::size_t size, std::size_t count,
                    void *context) {
  auto &bytes = *static_cast<std::vector<unsigned char> *>(context);
  const std::size_t length = size * count;
  if (length > max_tile_bytes - bytes.size())
    return 0;
  bytes.insert(bytes.end(), contents, contents + length);
  return length;
}

int cancel_request(void *context, curl_off_t, curl_off_t, curl_off_t,
                   curl_off_t) {
  return static_cast<const std::atomic<bool> *>(context)->load() ? 1 : 0;
}

std::string tile_url(TileKey key, bool elevation) {
  const auto level = std::to_string(key.level);
  const auto x = std::to_string(key.x);
  const auto y = std::to_string(key.y);
  if (const char *base = std::getenv("ATLAS_TILE_BASE_URL"))
    return std::string(base) + (elevation ? "/elevation/" : "/imagery/") +
           level + "/" + x + "/" + y + (elevation ? ".png" : ".jpg");
  if (elevation)
    return "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/" + level +
           "/" + x + "/" + y + ".png";
  return "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/g/" +
         level + "/" + y + "/" + x + ".jpg";
}

void allocate_texture(GLuint name, int size, GLenum format, GLint filter) {
  glBindTexture(GL_TEXTURE_2D, name);
  glTexImage2D(GL_TEXTURE_2D, 0, format, size, size, 0, format,
               GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

} // namespace

struct EarthTiles::State {
  explicit State(const fs::path &executable, bool stream)
      : bundled{executable.parent_path() / "earth-detail",
                executable.parent_path().parent_path() / "Resources" /
                    "earth-detail",
                fs::path("assets") / "earth-detail"} {
    GLint limit = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &limit);
    if (limit < atlas_size)
      return;
    glGenTextures(static_cast<GLsizei>(textures.size()), textures.data());
    allocate_texture(textures[0], atlas_size, rgb, GL_LINEAR);
    allocate_texture(textures[1], atlas_size, rgb, GL_LINEAR);
    allocate_texture(textures[2], grid_size, GL_RGBA, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
    if (glGetError() != 0) {
      glDeleteTextures(static_cast<GLsizei>(textures.size()), textures.data());
      textures.fill(0);
      return;
    }

    online = stream && curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK;
    std::error_code error;
    const auto temp = fs::temp_directory_path(error);
    if (!error) {
      cache = temp / "wildlife-atlas-earth-tiles-v1";
      if (const char *base = std::getenv("ATLAS_TILE_BASE_URL"))
        cache = temp / ("wildlife-atlas-custom-" +
                        std::to_string(std::hash<std::string>{}(base)));
      if (const char *override_path = std::getenv("ATLAS_TILE_CACHE_DIR"))
        cache = override_path;
      fs::create_directories(cache, error);
      if (error)
        cache.clear();
      for (fs::directory_iterator it(cache, error), end; !error && it != end;
           it.increment(error)) {
        if (it->is_regular_file(error))
          disk_bytes += it->file_size(error);
      }
    }
    const int worker_count =
        std::getenv("ATLAS_TILE_WORKERS")
            ? std::clamp(std::atoi(std::getenv("ATLAS_TILE_WORKERS")), 1, 4)
            : 4;
    for (int i = 0; i < worker_count; ++i)
      workers.emplace_back([this] { load(); });
  }

  ~State() {
    {
      std::lock_guard lock(mutex);
      stopping = true;
    }
    wake.notify_all();
    for (auto &worker : workers)
      if (worker.joinable())
        worker.join();
    if (online)
      curl_global_cleanup();
    glDeleteTextures(static_cast<GLsizei>(textures.size()), textures.data());
  }

  void load() {
    using CurlHandle = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>;
    CurlHandle client(online ? curl_easy_init() : nullptr, curl_easy_cleanup);
    if (client) {
      curl_easy_setopt(client.get(), CURLOPT_NOSIGNAL, 1L);
      curl_easy_setopt(client.get(), CURLOPT_FAILONERROR, 1L);
      curl_easy_setopt(client.get(), CURLOPT_CONNECTTIMEOUT_MS, 2400L);
      curl_easy_setopt(client.get(), CURLOPT_TIMEOUT_MS, 4000L);
      curl_easy_setopt(client.get(), CURLOPT_USERAGENT,
                       "WildlifeAtlas/0.1 (interactive Earth terrain)");
      curl_easy_setopt(client.get(), CURLOPT_WRITEFUNCTION, receive);
      curl_easy_setopt(client.get(), CURLOPT_NOPROGRESS, 0L);
      curl_easy_setopt(client.get(), CURLOPT_XFERINFOFUNCTION, cancel_request);
      curl_easy_setopt(client.get(), CURLOPT_XFERINFODATA, &stopping);
    }

    while (!stopping) {
      TileRequest request;
      {
        std::unique_lock lock(mutex);
        wake.wait(lock, [&] { return stopping || !queue.empty(); });
        if (stopping)
          break;
        request = queue.front();
        queue.erase(queue.begin());
        active.insert(request);
      }
      struct Progress {
        State *state;
        TileRequest request;
      } progress{this, request};
      if (client) {
        curl_easy_setopt(client.get(), CURLOPT_XFERINFODATA, &progress);
        curl_easy_setopt(
            client.get(), CURLOPT_XFERINFOFUNCTION,
            +[](void *data, curl_off_t, curl_off_t, curl_off_t,
                curl_off_t) -> int {
              const auto &p = *static_cast<Progress *>(data);
              if (p.state->stopping)
                return 1;
              std::lock_guard lock(p.state->mutex);
              return p.state->wanted.contains(p.request) ? 0 : 1;
            });
      }
      Tile tile{request.key, {}, {}, request.elevation};
      (request.elevation ? tile.elevation : tile.photo) =
          fetch(request.key, request.elevation, client.get());
      {
        std::lock_guard lock(mutex);
        if (wanted.contains(request))
          completed.push_back(std::move(tile));
        active.erase(request);
      }
    }
  }

  std::vector<unsigned char> fetch(TileKey key, bool elevation, CURL *client) {
    const std::string name =
        std::to_string(key.level) + "_" + std::to_string(key.x) + "_" +
        std::to_string(key.y) + (elevation ? ".png" : ".jpg");
    for (const auto &directory : bundled) {
      if (auto pixels = decode(read_bytes(directory / name)); !pixels.empty())
        return pixels;
    }
    if (!cache.empty()) {
      if (auto pixels = decode(read_bytes(cache / name)); !pixels.empty())
        return pixels;
    }
    if (!client || stopping)
      return {};

    // Limit the cadence for the public imagery service, even when a view moves
    // quickly.
    if (!elevation) {
      Clock::time_point next;
      {
        std::lock_guard lock(photo_mutex);
        next = std::max(next_photo, Clock::now());
        next_photo = next + 100ms;
      }
      std::this_thread::sleep_until(next);
    }
    if (stopping)
      return {};
    std::vector<unsigned char> bytes;
    const auto url = tile_url(key, elevation);
    curl_easy_setopt(client, CURLOPT_URL, url.c_str());
    curl_easy_setopt(client, CURLOPT_WRITEDATA, &bytes);
    if (curl_easy_perform(client) != CURLE_OK)
      return {};
    long response = 0;
    curl_easy_getinfo(client, CURLINFO_RESPONSE_CODE, &response);
    if (response != 200)
      return {};
    auto pixels = decode(bytes);
    if (!pixels.empty() && !cache.empty()) {
      const auto reserved = disk_bytes.fetch_add(bytes.size());
      if (reserved + bytes.size() > disk_budget) {
        disk_bytes.fetch_sub(bytes.size());
        return pixels;
      }
      const auto path = cache / name;
      auto temporary = path;
      temporary += "." +
                   std::to_string(Clock::now().time_since_epoch().count()) +
                   ".part";
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      output.write(reinterpret_cast<const char *>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
      output.close();
      std::error_code error;
      if (output) {
        fs::rename(temporary, path, error);
        if (error)
          disk_bytes.fetch_sub(bytes.size());
      }
      if (!output)
        disk_bytes.fetch_sub(bytes.size());
      fs::remove(temporary, error);
    }
    return pixels;
  }

  int slot_for(TileKey key) const {
    const auto match =
        std::find_if(slots.begin(), slots.end(),
                     [&](const Slot &slot) { return slot.key == key; });
    return match == slots.end() ? -1 : static_cast<int>(match - slots.begin());
  }

  void apply_completed() {
    std::vector<Tile> tiles;
    {
      std::lock_guard lock(mutex);
      tiles.swap(completed);
    }
    for (auto &tile : tiles) {
      const auto key = tile.key;
      int index = slot_for(key);
      if (tile.photo.empty() && tile.elevation.empty()) {
        retry[{key, tile.elevation_request}] = Clock::now() + 30s;
        continue;
      }
      if (index < 0) {
        auto oldest = std::min_element(
            slots.begin(), slots.end(),
            [](const Slot &a, const Slot &b) { return a.age < b.age; });
        index = static_cast<int>(oldest - slots.begin());
        *oldest = {key, age, 0, Clock::now(), {}, {}};
      }
      auto &slot = slots[index];
      slot.age = age;
      const int x = (index % grid_size) * tile_size;
      const int y = (index / grid_size) * tile_size;
      const auto upload = [&](int layer,
                              const std::vector<unsigned char> &pixels) {
        if (pixels.empty())
          return;
        glBindTexture(GL_TEXTURE_2D, textures[layer]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, tile_size, tile_size, rgb,
                        GL_UNSIGNED_BYTE, pixels.data());
        slot.layers |= 1 << layer;
      };
      upload(0, tile.photo);
      upload(1, tile.elevation);
      if (!tile.photo.empty())
        slot.photo = std::move(tile.photo);
      if (!tile.elevation.empty())
        slot.elevation = std::move(tile.elevation);
      retry.erase({key, tile.elevation_request});
    }
  }

  void request(std::vector<TileRequest> keys) {
    {
      std::lock_guard lock(mutex);
      queue.clear();
      wanted.clear();
      wanted.insert(keys.begin(), keys.end());
      std::set<TileRequest> queued;
      for (auto key : keys) {
        if (!queued.insert(key).second)
          continue;
        if (active.contains(key) ||
            std::any_of(completed.begin(), completed.end(),
                        [&](const Tile &tile) {
                          return tile.key == key.key &&
                                 tile.elevation_request == key.elevation;
                        }))
          continue;
        queue.push_back(key);
      }
    }
    if (!keys.empty())
      wake.notify_all();
  }

  const std::array<fs::path, 3> bundled;
  std::array<GLuint, 3> textures{};
  std::array<Slot, grid_size * grid_size> slots{};
  std::array<unsigned char, grid_size * grid_size * 4> last_page{};
  std::uint64_t age = 0;
  std::uint64_t revision = 0;
  bool online = false;
  std::vector<std::thread> workers;
  std::atomic<bool> stopping{false};
  std::mutex mutex;
  std::condition_variable wake;
  std::vector<TileRequest> queue;
  std::set<TileRequest> active, wanted;
  std::vector<Tile> completed;
  std::map<TileRequest, Clock::time_point> retry;
  fs::path cache;
  std::atomic<std::uintmax_t> disk_bytes{0};
  std::mutex photo_mutex;
  Clock::time_point next_photo{};
};

EarthTiles::EarthTiles(const fs::path &executable, bool stream)
    : state_(std::make_unique<State>(executable, stream)) {}
EarthTiles::~EarthTiles() = default;

EarthTileView EarthTiles::update(ImVec2 center, float radius, ImVec2 min,
                                 ImVec2 max, Vec3 east, Vec3 north,
                                 Vec3 forward) {
  auto &state = *state_;
  if (!state.textures[0])
    return {};
  const auto window =
      choose_window(center, radius, min, max, east, north, forward);
  if (!window) {
    state.request({});
    return {};
  }

  GLint previous = 0;
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous);
  ++state.age;
  state.apply_completed();

  const int side = 1 << window->level;
  const auto now = Clock::now();
  for (auto it = state.retry.begin(); it != state.retry.end();) {
    if (it->second <= now)
      it = state.retry.erase(it);
    else
      ++it;
  }
  std::vector<std::pair<double, TileRequest>> wanted;
  wanted.reserve(grid_size * grid_size);
  decltype(state.last_page) page{};
  bool ready = false;
  for (int y = window->top; y <= window->bottom; ++y) {
    for (int x = window->left; x <= window->right; ++x) {
      const TileKey key{window->level, (x + side) % side, y};
      int slot = state.slot_for(key);
      const auto child_layers = slot < 0 ? 0 : state.slots[slot].layers;
      if (!(child_layers & 1) && key.level > min_zoom) {
        const TileKey parent{key.level - 1, key.x / 2, key.y / 2};
        const int coarse = state.slot_for(parent);
        const auto layers = coarse < 0 ? 0 : state.slots[coarse].layers;
        const double priority = -100 + std::hypot(x + .5 - window->center_x,
                                                  y + .5 - window->center_y);
        if (!(layers & 1) && !state.retry.contains({parent, false}))
          wanted.emplace_back(priority, TileRequest{parent, false});
        if (!(layers & 2) && !(child_layers & 2) &&
            !state.retry.contains({parent, true}))
          wanted.emplace_back(priority + .25, TileRequest{parent, true});
      }
      if (slot < 0 || state.slots[slot].layers != 3) {
        const double distance =
            std::hypot(x + 0.5 - window->center_x, y + 0.5 - window->center_y);
        const auto layers = slot < 0 ? 0 : state.slots[slot].layers;
        if (!(layers & 1) && !state.retry.contains({key, false}))
          wanted.emplace_back(distance, TileRequest{key, false});
        if (!(layers & 2) && !state.retry.contains({key, true}))
          wanted.emplace_back(distance + .25, TileRequest{key, true});
      }
      // A newly arrived fine elevation tile must not hide already available
      // coarse imagery. Prefer the finest imagery, then any elevation-only
      // tile.
      auto parent = key, fallback_key = key;
      int fallback_slot = -1;
      slot = -1;
      for (;;) {
        const int candidate = state.slot_for(parent);
        if (candidate >= 0) {
          if (fallback_slot < 0) {
            fallback_slot = candidate;
            fallback_key = parent;
          }
          if (state.slots[candidate].layers & 1) {
            slot = candidate;
            break;
          }
        }
        if (parent.level <= min_zoom)
          break;
        parent = {parent.level - 1, parent.x / 2, parent.y / 2};
      }
      if (slot < 0) {
        slot = fallback_slot;
        parent = fallback_key;
      }
      if (slot < 0)
        continue;
      state.slots[slot].age = state.age;
      const auto offset =
          ((y - window->top) * grid_size + x - window->left) * 4;
      page[offset] = static_cast<unsigned char>(1 + slot % grid_size);
      page[offset + 1] = static_cast<unsigned char>(1 + slot / grid_size);
      page[offset + 2] = static_cast<unsigned char>(key.level - parent.level);
      const double fade =
          std::chrono::duration<double>(now - state.slots[slot].arrival)
              .count() /
          0.24;
      const auto opacity =
          static_cast<unsigned char>(std::clamp(fade, 0.0, 1.0) * 63.0);
      page[offset + 3] =
          static_cast<unsigned char>(state.slots[slot].layers | (opacity << 2));
      ready = true;
    }
  }
  std::sort(wanted.begin(), wanted.end());
  std::vector<TileRequest> keys;
  keys.reserve(wanted.size());
  for (const auto &entry : wanted)
    keys.push_back(entry.second);
  state.request(std::move(keys));

  if (page != state.last_page) {
    glBindTexture(GL_TEXTURE_2D, state.textures[2]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, grid_size, grid_size, GL_RGBA,
                    GL_UNSIGNED_BYTE, page.data());
    state.last_page = page;
    ++state.revision;
  }
  glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(previous));
  if (!ready)
    return {};
  return {static_cast<float>(window->left), static_cast<float>(window->top),
          static_cast<float>(side)};
}

float EarthTiles::elevation(EarthTileView view, double u, double sin_latitude,
                            float fallback, bool land_only) const {
  if (view.side == 0.0f || std::abs(sin_latitude) >= mercator_limit)
    return fallback;
  double x = u * view.side;
  x += view.side * std::round((view.x + grid_size * 0.5 - x) / view.side);
  const double y = mercator_y(sin_latitude) * view.side;
  const int col = static_cast<int>(std::floor(x - view.x));
  const int row = static_cast<int>(std::floor(y - view.y));
  if (col < -1 || col > grid_size || row < -1 || row > grid_size)
    return fallback;

  const auto sample = [&](int c, int r) {
    if (c < 0 || c >= grid_size || r < 0 || r >= grid_size)
      return fallback;
    const auto *page = &state_->last_page[4 * (r * grid_size + c)];
    if (page[0] == 0 || !(page[3] & 2))
      return fallback;
    const auto &pixels =
        state_->slots[(page[1] - 1) * grid_size + page[0] - 1].elevation;
    const double scale = static_cast<double>(1 << page[2]);
    const auto texel = [scale](double tile, double start) {
      const double uv = tile / scale - std::floor(start / scale);
      return std::clamp(uv * tile_size - 0.5, 0.0, tile_size - 1.0);
    };
    const double px = texel(x, c + view.x), py = texel(y, r + view.y);
    const int x0 = static_cast<int>(px), y0 = static_cast<int>(py);
    const int x1 = std::min(x0 + 1, tile_size - 1),
              y1 = std::min(y0 + 1, tile_size - 1);
    const auto height = [&](int tx, int ty) {
      const auto *rgb = &pixels[3 * (ty * tile_size + tx)];
      return (256 * rgb[0] + rgb[1] - 32768) + rgb[2] / 256.0f;
    };
    const auto meters =
        std::lerp(std::lerp(height(x0, y0), height(x1, y0), px - x0),
                  std::lerp(height(x0, y1), height(x1, y1), px - x0), py - y0);
    return std::lerp(fallback,
                     static_cast<float>(std::clamp(
                         meters, land_only ? 0.0 : -11000.0, 9000.0)),
                     (page[3] >> 2) / 63.0f);
  };

  // Overlap adjacent pages as they load so an incomplete tile set cannot form
  // cliffs.
  const auto weight = [](double fraction) {
    const double edge = std::min(fraction, 1.0 - fraction);
    const double t = std::clamp(edge / (10.0 / tile_size), 0.0, 1.0);
    return 0.5 * (1.0 - t * t * (3.0 - 2.0 * t));
  };
  const double fx = x - std::floor(x), fy = y - std::floor(y);
  const double wx = weight(fx), wy = weight(fy);
  const int next_col = col + (fx < 0.5 ? -1 : 1),
            next_row = row + (fy < 0.5 ? -1 : 1);
  const auto row_height = [&](int r) {
    const double height = sample(col, r);
    return wx > 0.0 ? std::lerp(height, sample(next_col, r), wx) : height;
  };
  const double height = row_height(row);
  return static_cast<float>(
      wy > 0.0 ? std::lerp(height, row_height(next_row), wy) : height);
}

Vec3 EarthTiles::color(EarthTileView view, double u, double sin_latitude,
                       Vec3 fallback) const {
  if (view.side == 0.0f || std::abs(sin_latitude) >= mercator_limit)
    return fallback;
  double x = u * view.side;
  x += view.side * std::round((view.x + grid_size * .5 - x) / view.side);
  const double y = mercator_y(sin_latitude) * view.side;
  const int col = static_cast<int>(std::floor(x - view.x)),
            row = static_cast<int>(std::floor(y - view.y));
  if (col < 0 || col >= grid_size || row < 0 || row >= grid_size)
    return fallback;
  const auto *page = &state_->last_page[4 * (row * grid_size + col)];
  if (!page[0] || !(page[3] & 1))
    return fallback;
  const auto &pixels =
      state_->slots[(page[1] - 1) * grid_size + page[0] - 1].photo;
  const double scale = static_cast<double>(1 << page[2]);
  const auto texel = [scale](double tile, double start) {
    return std::clamp((tile / scale - std::floor(start / scale)) * tile_size -
                          0.5,
                      0.0, tile_size - 1.0);
  };
  const double px = texel(x, col + view.x), py = texel(y, row + view.y);
  const int x0 = static_cast<int>(px), y0 = static_cast<int>(py);
  const int x1 = std::min(x0 + 1, tile_size - 1),
            y1 = std::min(y0 + 1, tile_size - 1);
  const float fade = (page[3] >> 2) / 63.0f;
  const auto channel = [&](int ch, float base) {
    const auto at = [&](int c, int r) {
      return pixels[3 * (r * tile_size + c) + ch] / 255.0;
    };
    const auto value =
        std::lerp(std::lerp(at(x0, y0), at(x1, y0), px - x0),
                  std::lerp(at(x0, y1), at(x1, y1), px - x0), py - y0);
    return std::lerp(base, static_cast<float>(value), fade);
  };
  return {channel(0, fallback.x), channel(1, fallback.y),
          channel(2, fallback.z)};
}

std::uint64_t EarthTiles::revision() const { return state_->revision; }

void EarthTiles::bind(unsigned int fallback_texture) const {
  for (std::size_t i = 0; i < state_->textures.size(); ++i) {
    glActiveTexture(static_cast<GLenum>(GL_TEXTURE0 + 3 + i));
    glBindTexture(GL_TEXTURE_2D,
                  state_->textures[i] ? state_->textures[i] : fallback_texture);
  }
  glActiveTexture(GL_TEXTURE0);
}

} // namespace wildlife
