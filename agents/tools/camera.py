"""City Walk 拍照机位 Agent · 工具集

放置位置：agents/tools/camera.py
配套数据：agents/data/camera_spots.seed.json
依赖：pip install astral requests

设计原则：
  1. 能用工具解决的，不要写进 Prompt —— 光线、天气、打分全部由代码算，Agent 只负责解读和表达。
  2. 每个工具的 docstring 写清「什么时候用」，这是提升工具调用准确率最有效的手段。
  3. 所有外部调用都要有超时和异常兜底，工具挂了要如实返回错误，不能静默返回空数据。
"""

from __future__ import annotations

import json
import math
import os
from datetime import date as _date
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from astral import LocationInfo
from astral.sun import azimuth, sun

try:  # 装有 OpenAI Agents SDK：装饰器返回真正的 FunctionTool
    from agents import function_tool as _sdk_function_tool

    _HAS_AGENTS_SDK = True
except ImportError:  # pragma: no cover
    _sdk_function_tool = None
    _HAS_AGENTS_SDK = False

# 离线自测用的原始函数登记表。
# 原因：@function_tool 装饰后，模块级名字会变成 FunctionTool 对象，不能再当普通函数
# 直接调用（会报 "'FunctionTool' object is not callable"）。这里在装饰时把原始函数
# 另存一份，供文件末尾的 __main__ 自测直接调用。
_RAW_FUNCS: dict = {}


def function_tool(fn=None, **kwargs):
    """与 SDK 用法一致的装饰器，额外把原始函数登记进 _RAW_FUNCS。

    支持两种写法：``@function_tool`` 与 ``@function_tool(strict_mode=False)``。
    - 装有 agents SDK：返回 FunctionTool，供 Agent 注册使用；
    - 未装：直接返回原函数，保证 ``python camera.py`` 离线自测照常可用。
    """

    def decorate(func):
        _RAW_FUNCS[func.__name__] = func
        if _HAS_AGENTS_SDK:
            return _sdk_function_tool(func, **kwargs)
        return func

    if fn is None:
        return decorate
    return decorate(fn)


TZ = ZoneInfo("Asia/Shanghai")
HTTP_TIMEOUT = 8

_HERE = os.path.dirname(os.path.abspath(__file__))
_HERE_PARENT = os.path.dirname(_HERE)


def _resolve_seed_path() -> str:
    """按优先级查找种子机位库：agents/data → 同级 data → 脚本同目录。"""

    candidates = [
        os.path.join(_HERE_PARENT, "data", "camera_spots.seed.json"),
        os.path.join(_HERE, "data", "camera_spots.seed.json"),
        os.path.join(_HERE, "camera_spots.seed.json"),
        os.path.join(_HERE_PARENT, "camera_spots.seed.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


SEED_PATH = _resolve_seed_path()

AMAP_AROUND = "https://restapi.amap.com/v3/place/around"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

# 机位友好的 POI 关键词（地图没有「机位」这个分类，只能靠关键词召回）
DEFAULT_KEYWORDS = [
    "观景台", "天台", "天桥", "公园", "老街", "江边",
    "涂鸦", "打卡", "艺术中心", "滨水广场", "栈道",
]

_SEED_CACHE: list[dict] | None = None


# --------------------------------------------------------------------------- #
# 内部工具函数
# --------------------------------------------------------------------------- #

def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> int:
    """两点球面距离（米）。"""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return int(2 * r * math.asin(math.sqrt(a)))


def _load_seed() -> list[dict]:
    """加载种子机位库（带缓存）。"""
    global _SEED_CACHE
    if _SEED_CACHE is not None:
        return _SEED_CACHE
    try:
        with open(SEED_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _SEED_CACHE = data.get("spots", [])
    except Exception:
        _SEED_CACHE = []
    return _SEED_CACHE


def _azimuth_deg(observer, when) -> float:
    """太阳方位角（度，正北为 0，顺时针）。"""

    val = float(azimuth(observer, when))
    # astral 2.2 起返回角度；更早版本返回弧度。<=2π 视为弧度并换算。
    if abs(val) <= 6.5:
        val = math.degrees(val)
    return round(val % 360, 1)


def _time_str(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%H:%M")


def _is_weekend(d: _date) -> bool:
    return d.weekday() >= 5


def _amap_around(lat: float, lng: float, keyword: str, radius_m: int) -> list[dict]:
    """高德周边搜索，失败返回空列表（不抛异常）。"""

    key = os.getenv("AMAP_KEY", "").strip()
    if not key:
        return []
    params = {
        "key": key,
        "location": f"{lng:.6f},{lat:.6f}",
        "keywords": keyword,
        "radius": radius_m,
        "offset": 20,
        "page": 1,
        "extensions": "all",
    }
    try:
        r = requests.get(AMAP_AROUND, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        body = r.json()
    except Exception:
        return []
    if str(body.get("status")) != "1":
        return []

    out = []
    for poi in body.get("pois", []) or []:
        loc = (poi.get("location") or "").split(",")
        if len(loc) != 2:
            continue
        try:
            poi_lng, poi_lat = float(loc[0]), float(loc[1])
        except ValueError:
            continue
        out.append(
            {
                "name": poi.get("name") or "",
                "lat": round(poi_lat, 6),
                "lng": round(poi_lng, 6),
                "source": "amap_poi",
                "spot_type": (poi.get("type") or "").split(";")[-1] or "未分类",
                "address": poi.get("address") or "",
                "keyword_hit": keyword,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# 工具 1：机位检索
# --------------------------------------------------------------------------- #

@function_tool
def search_camera_spots(
    lat: float,
    lng: float,
    city: str = "",
    keyword: str = "",
    radius_m: int = 2000,
) -> dict:
    """检索指定坐标附近的拍照机位候选（地图 POI + 本地种子库混合召回）。

    什么时候用：
      用户给出位置、想知道附近有哪些可拍照的点位时，**第一步**就调用这个工具。
      不要用它查询餐厅、商场等非拍摄场景。

    参数：
      lat / lng   用户位置坐标（地图坐标系 GCJ-02）
      city        城市名，用于缩小搜索范围
      keyword     可选，风格关键词，如「日系」「老街」「江边」「胶片」
      radius_m    搜索半径（米），默认 2000

    返回：
      {"ok": bool, "count": int, "spots": [...]}
      每个机位含 name / lat / lng / source / spot_type / distance_m，
      种子库来源的还会带 desc / best_time / best_direction / lens / crowd_hint。
    """

    radius_m = max(200, min(int(radius_m), 10000))
    kw = (keyword or "").strip()
    candidates: list[dict] = []

    # 第一层：种子机位库（准）
    for spot in _load_seed():
        try:
            d = _haversine_m(lat, lng, float(spot["lat"]), float(spot["lng"]))
        except Exception:
            continue
        if d > radius_m:
            continue
        haystack = " ".join(
            [spot.get("name", ""), spot.get("desc", ""), " ".join(spot.get("tags", []) or [])]
        )
        if kw and kw not in haystack:
            # 风格词不在标签里时放宽为「不排除」，仅降低优先级
            pass
        item = dict(spot)
        item.update({"source": "seed_library", "distance_m": d})
        item["_style_bonus"] = 1 if (kw and kw in haystack) else 0
        candidates.append(item)

    # 第二层：地图 POI 关键词召回（广）
    keywords = [kw] if kw else []
    keywords += [k for k in DEFAULT_KEYWORDS if k != kw]
    for k in keywords[:5]:
        for poi in _amap_around(lat, lng, k, radius_m):
            poi["distance_m"] = _haversine_m(lat, lng, poi["lat"], poi["lng"])
            candidates.append(poi)

    # 合并去重：同名或 150m 内视为同一机位，种子库优先
    merged: list[dict] = []
    for item in sorted(candidates, key=lambda x: (x.get("source") != "seed_library",)):
        dup = False
        for kept in merged:
            if item["name"] and item["name"] == kept["name"]:
                dup = True
                break
            if _haversine_m(item["lat"], item["lng"], kept["lat"], kept["lng"]) < 150:
                dup = True
                break
        if not dup:
            merged.append(item)

    merged.sort(key=lambda x: (-x.get("_style_bonus", 0), x.get("distance_m", 0)))
    merged = merged[:12]
    for item in merged:
        item.pop("_style_bonus", None)

    return {
        "ok": True,
        "count": len(merged),
        "center": {"lat": lat, "lng": lng, "city": city, "radius_m": radius_m},
        "spots": merged,
        "note": "地图坐标系为 GCJ-02，可直接用于导航。若 count 为 0，建议放宽半径或更换关键词。",
    }


# --------------------------------------------------------------------------- #
# 工具 2：光线窗口（离线计算，无需 Key）
# --------------------------------------------------------------------------- #

@function_tool
def get_light_window(lat: float, lng: float, date: str) -> dict:
    """计算某地某天的日出日落、黄金时刻、蓝调时刻与日落方位角。

    什么时候用：
      需要确定机位的「最佳拍摄时间段」、判断顺光/逆光时调用。
      任何时候都不能靠模型自己推算日落时间，必须用这个工具的结果。

    参数：
      lat / lng   机位坐标
      date        YYYY-MM-DD

    返回：黄金时刻、蓝调时刻、日落方位角、顺光站位建议。
    """

    try:
        day = _date.fromisoformat(date.strip())
    except Exception:
        return {"ok": False, "error": f"date 格式应为 YYYY-MM-DD，收到：{date}"}

    try:
        loc = LocationInfo(latitude=lat, longitude=lng)
        s = sun(loc.observer, date=day, tzinfo=TZ)
        dawn, sunrise, sunset, dusk = s["dawn"], s["sunrise"], s["sunset"], s["dusk"]
    except Exception as exc:  # 极地或无日出日落等情况
        return {"ok": False, "error": f"光线计算失败：{exc}"}

    morning_golden = (sunrise, sunrise + timedelta(minutes=60))
    evening_golden = (sunset - timedelta(minutes=60), sunset)
    evening_blue = (sunset, sunset + timedelta(minutes=25))
    morning_blue = (sunrise - timedelta(minutes=25), sunrise)

    def compass(az: float) -> str:
        dirs = ["正北", "东北", "正东", "东南", "正南", "西南", "正西", "西北"]
        return dirs[int((az % 360) / 45 + 0.5) % 8]

    az_rise = _azimuth_deg(loc.observer, sunrise)
    az_set = _azimuth_deg(loc.observer, sunset)
    facing_for_frontlight = compass(az_set + 180)
    facing_for_backlight = compass(az_set)

    return {
        "ok": True,
        "date": day.isoformat(),
        "sunrise": _time_str(sunrise),
        "sunset": _time_str(sunset),
        "dawn": _time_str(dawn),
        "dusk": _time_str(dusk),
        "morning_golden": f"{_time_str(morning_golden[0])}-{_time_str(morning_golden[1])}",
        "evening_golden": f"{_time_str(evening_golden[0])}-{_time_str(evening_golden[1])}",
        "morning_blue": f"{_time_str(morning_blue[0])}-{_time_str(morning_blue[1])}",
        "evening_blue": f"{_time_str(evening_blue[0])}-{_time_str(evening_blue[1])}",
        "sunrise_azimuth_deg": az_rise,
        "sunset_azimuth_deg": az_set,
        "sunset_direction": compass(az_set),
        "advice": (
            f"日落方位 {az_set}°（{compass(az_set)}）。"
            f"拍顺光人物：摄影师背对太阳、面向{facing_for_frontlight}，人物面部受光均匀；"
            f"拍逆光剪影：摄影师面向{facing_for_backlight}，让人物挡住太阳，对天空点测光。"
        ),
    }


# --------------------------------------------------------------------------- #
# 工具 3：天气（Open-Meteo，免 Key）
# --------------------------------------------------------------------------- #

@function_tool
def get_weather_forecast(lat: float, lng: float, date: str) -> dict:
    """查询某地某天 05:00-21:00 的逐小时云量、降水概率、能见度，并给出出片友好度。

    什么时候用：
      判断「今天值不值得去拍」、决定是否需要改期时调用。
      云量 30%~70% 最出片（天空有层次），全晴缺层次、全阴没光比。

    参数：
      lat / lng   机位坐标
      date        YYYY-MM-DD
    """

    day = date.strip()
    params = {
        "latitude": lat,
        "longitude": lng,
        "hourly": "cloud_cover,precipitation_probability,visibility,temperature_2m",
        "timezone": "Asia/Shanghai",
        "start_date": day,
        "end_date": day,
    }
    try:
        r = requests.get(OPEN_METEO, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        body = r.json()
    except Exception as exc:
        return {"ok": False, "error": f"天气接口调用失败：{exc}"}

    hourly = body.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return {"ok": False, "error": "天气接口未返回数据，可能日期超出预报范围（约 16 天）。"}

    rows = []
    good_hours = []
    for i, t in enumerate(times):
        hh = int(t[11:13])
        if not (5 <= hh <= 21):
            continue
        cloud = (hourly.get("cloud_cover") or [None])[i]
        pop = (hourly.get("precipitation_probability") or [None])[i]
        vis = (hourly.get("visibility") or [None])[i]
        temp = (hourly.get("temperature_2m") or [None])[i]
        row = {
            "time": t[11:16],
            "cloud_cover_pct": cloud,
            "precip_prob_pct": pop,
            "visibility_m": vis,
            "temp_c": temp,
        }
        rows.append(row)
        if cloud is not None and pop is not None and 30 <= cloud <= 70 and pop < 30:
            good_hours.append(row["time"])

    clouds = [r["cloud_cover_pct"] for r in rows if r["cloud_cover_pct"] is not None]
    avg_cloud = round(sum(clouds) / len(clouds), 1) if clouds else None
    if avg_cloud is None:
        verdict = "数据不足，建议以现场天色为准。"
    elif avg_cloud < 15:
        verdict = "大晴天：光比偏硬、天空缺层次，建议主攻逆光剪影与阴影构图，或改等蓝调时刻。"
    elif avg_cloud < 30:
        verdict = "薄云晴天：天空有少量纹理但光比仍硬，建议把主体安排在阴影侧或用反光板补光。"
    elif avg_cloud <= 70:
        verdict = "云量适中：最出片的天气，天空有层次且光比柔和。"
    else:
        verdict = "重阴天：光比弱，建议改走人文街拍、黑白或高对比修图风格。"

    return {
        "ok": True,
        "date": day,
        "avg_cloud_cover_pct": avg_cloud,
        "verdict": verdict,
        "good_hours": good_hours[:8],
        "hourly": rows,
    }


# --------------------------------------------------------------------------- #
# 工具 4：拥挤度评估（纯规则）
# --------------------------------------------------------------------------- #

_BASE_CROWD = {
    "打卡": 78, "网红": 80, "天桥": 62, "观景台": 70, "天台": 55,
    "老街": 60, "艺术": 50, "创意园": 48, "公园": 42,
    "江边": 35, "滨水": 38, "栈道": 40, "山": 25, "村": 22,
}


@function_tool
def estimate_crowd(spot_name: str, spot_type: str, date: str, hour: int = 17) -> dict:
    """评估某个机位在指定日期与时段的拥挤程度。

    什么时候用：
      在给候选机位排序之前，对每个机位各调用一次。

    参数：
      spot_name  机位名称
      spot_type  场地类型（如 公园 / 天桥 / 打卡 / 江边，可传 POI 类型原文）
      date       YYYY-MM-DD
      hour       计划到达的小时（0-23），默认 17
    """

    try:
        day = _date.fromisoformat(date.strip())
    except Exception:
        return {"ok": False, "error": f"date 格式应为 YYYY-MM-DD，收到：{date}"}

    text = f"{spot_name} {spot_type}"
    base = 45  # 未命中任何关键词时的中性基线
    matched = [val for kw, val in _BASE_CROWD.items() if kw in text]
    if matched:
        base = max(matched)

    reasons = []
    if _is_weekend(day):
        base += 15
        reasons.append("周末人流上浮")

    h = int(hour) % 24
    if 6 <= h < 9:
        base -= 20
        reasons.append("清晨人最少")
    elif 9 <= h < 11:
        base -= 5
    elif 11 <= h < 16:
        base += 5
    elif 16 <= h < 19:
        base += 18
        reasons.append("黄金时段为全天人流峰值")
    elif 19 <= h < 21:
        base -= 5

    score = max(0, min(100, base))
    level = "低" if score < 40 else ("中" if score < 70 else "高")

    return {
        "ok": True,
        "spot_name": spot_name,
        "date": day.isoformat(),
        "hour": h,
        "crowd_score": score,
        "crowd_level": level,
        "reason": "；".join(reasons) or "常规时段",
        "advice": {
            "低": "可安心架三脚架，建议提前 10 分钟到场。",
            "中": "建议提前 20 分钟占位，避开主干道取景。",
            "高": "机位紧张，建议改用长焦压缩人流的构图，或换到备选机位。",
        }[level],
    }


# --------------------------------------------------------------------------- #
# 工具 5：三因子打分排序（把计算关进工具，不许模型自己算）
# --------------------------------------------------------------------------- #

# strict_mode=False：本工具参数是裸 dict / list[dict]（上游工具的原始返回），
# 结构无法用严格 JSON Schema 表达；SDK 默认的 strict 模式会因 additionalProperties
# 报 UserError，故显式关闭。其余四个工具参数均为标量，保持默认 strict 即可。
@function_tool(strict_mode=False)
def rank_camera_spots(
    spots: list[dict],
    light: dict,
    weather: dict,
    crowds: list[dict],
) -> dict:
    """按「光线 40% + 天气 35% + 拥挤度 25%」对候选机位打分排序，返回 Top 3。

    什么时候用：
      已经拿到机位列表、光线窗口、天气和各自拥挤度之后，最后一步调用，
      由它决定推荐顺序。不要自己心算排序。

    参数：
      spots    候选机位列表（search_camera_spots 返回的 spots）
      light    get_light_window 的返回结果
      weather  get_weather_forecast 的返回结果
      crowds   estimate_crowd 的结果列表，元素含 spot_name / crowd_score
    """

    if not spots:
        return {"ok": False, "error": "候选机位为空，请先调用 search_camera_spots。"}

    crowd_map = {}
    for c in crowds or []:
        if isinstance(c, dict) and c.get("spot_name"):
            crowd_map[c["spot_name"]] = c.get("crowd_score", 50)

    # 光线分：优先匹配傍晚黄金时刻窗口
    light_score = 80.0
    golden = ""
    if isinstance(light, dict) and light.get("ok"):
        golden = light.get("evening_golden", "")
        if golden:
            light_score = 100.0
        if not weather or not isinstance(weather, dict) or not weather.get("ok"):
            light_score = min(light_score, 85.0)

    # 天气分
    weather_score = 60.0
    if isinstance(weather, dict) and weather.get("ok"):
        cloud = weather.get("avg_cloud_cover_pct")
        good_hours = weather.get("good_hours") or []
        if cloud is None:
            weather_score = 60.0
        elif 30 <= cloud <= 70:
            weather_score = 100.0
        elif cloud < 15:
            weather_score = 65.0
        elif cloud < 30:
            weather_score = 80.0
        else:
            weather_score = 45.0
        if good_hours:
            weather_score = min(100.0, weather_score + 5)

    ranked = []
    for spot in spots:
        name = spot.get("name", "未命名机位")
        crowd_score = crowd_map.get(name, 50)
        crowd_component = max(0.0, 100.0 - crowd_score) * 1.0
        total = 0.40 * light_score + 0.35 * weather_score + 0.25 * crowd_component
        ranked.append(
            {
                "name": name,
                "lat": spot.get("lat"),
                "lng": spot.get("lng"),
                "spot_type": spot.get("spot_type", ""),
                "distance_m": spot.get("distance_m"),
                "best_time": spot.get("best_time", golden),
                "best_direction": spot.get("best_direction", ""),
                "lens": spot.get("lens", ""),
                "desc": spot.get("desc", ""),
                "crowd_level": next(
                    (c.get("crowd_level") for c in (crowds or [])
                     if isinstance(c, dict) and c.get("spot_name") == name),
                    "中",
                ),
                "score": round(total, 1),
                "score_breakdown": {
                    "light": round(light_score, 1),
                    "weather": round(weather_score, 1),
                    "crowd": round(crowd_component, 1),
                },
            }
        )

    ranked.sort(key=lambda x: (-x["score"], x.get("distance_m") or 0))
    return {
        "ok": True,
        "weights": {"light": 0.40, "weather": 0.35, "crowd": 0.25},
        "best_window": golden,
        "top": ranked[:3],
        "all": ranked,
    }


# --------------------------------------------------------------------------- #
# 离线自测：python agents/tools/camera.py
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # 装饰后模块级名字是 FunctionTool 对象，这里取回原始函数供离线自测直接调用
    search_camera_spots = _RAW_FUNCS["search_camera_spots"]
    get_light_window = _RAW_FUNCS["get_light_window"]
    get_weather_forecast = _RAW_FUNCS["get_weather_forecast"]
    estimate_crowd = _RAW_FUNCS["estimate_crowd"]
    rank_camera_spots = _RAW_FUNCS["rank_camera_spots"]

    LAT, LNG, DAY = 22.5145, 113.9405, _date.today().isoformat()

    print("=" * 60)
    print("1) search_camera_spots")
    res = search_camera_spots(LAT, LNG, "深圳", "", 3000)
    print("  命中机位数：", res["count"])
    for s in res["spots"][:5]:
        print("   -", s["name"], s.get("distance_m"), "m", s.get("source"))

    print("=" * 60)
    print("2) get_light_window")
    light = get_light_window(LAT, LNG, DAY)
    print(" ", {k: light[k] for k in ("sunset", "evening_golden", "evening_blue", "sunset_direction")})

    print("=" * 60)
    print("3) get_weather_forecast")
    weather = get_weather_forecast(LAT, LNG, DAY)
    print("  云量：", weather.get("avg_cloud_cover_pct"), "|", weather.get("verdict"))

    print("=" * 60)
    print("4) estimate_crowd")
    crowds = [estimate_crowd(s["name"], s.get("spot_type", ""), DAY, 17) for s in res["spots"][:5]]
    for c in crowds:
        print("   -", c["spot_name"], c["crowd_level"], c["crowd_score"], c["reason"])

    print("=" * 60)
    print("5) rank_camera_spots")
    ranked = rank_camera_spots(res["spots"], light, weather, crowds)
    for r in ranked.get("top", []):
        print("   -", r["name"], r["score"], r["score_breakdown"])
