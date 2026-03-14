from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)
CORS(app)

# ── Train model on startup ──
print("Loading dataset and training Random Forest model...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'Crop_recommendation.csv'))

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, rf_model.predict(X_test)) * 100
print(f"Model ready — Accuracy: {accuracy:.2f}%")

CROP_META = {
    'rice':        {'emoji':'🌾','latin':'Oryza sativa',         'season':'Kharif · Jun–Nov',    'price':'₹18–22 / kg',  'priceKg':20,  'yieldStr':'2.5–3.5 t/ha','yieldMid':3.0,  'water':'Very High',   'tips':['Maintain 5–10 cm standing water during tillering','Transplant seedlings at 21–25 days for best establishment','Apply nitrogen in three split doses: basal, tillering, panicle initiation']},
    'maize':       {'emoji':'🌽','latin':'Zea mays',              'season':'Kharif · Jun–Oct',    'price':'₹15–20 / kg',  'priceKg':17,  'yieldStr':'4–7 t/ha',    'yieldMid':5.5,  'water':'Medium',      'tips':['Space plants at 60x20 cm for optimal canopy','Earth up at knee height to prevent lodging','Top-dress nitrogen at 30 and 60 days after sowing']},
    'chickpea':    {'emoji':'🫘','latin':'Cicer arietinum',       'season':'Rabi · Oct–Mar',      'price':'₹45–60 / kg',  'priceKg':52,  'yieldStr':'0.8–1.5 t/ha','yieldMid':1.1,  'water':'Low',         'tips':['Inoculate seeds with Rhizobium culture before sowing','Avoid irrigation after pod filling','Harvest at 60–70% pod browning to minimise losses']},
    'kidneybeans': {'emoji':'🫘','latin':'Phaseolus vulgaris',    'season':'Kharif / Rabi',       'price':'₹80–120 / kg', 'priceKg':100, 'yieldStr':'0.8–1.5 t/ha','yieldMid':1.1,  'water':'Low',         'tips':['Requires cool dry conditions — avoid waterlogged soils','Stake climbing varieties at 20 cm height','Foliar boron spray at flowering improves pod set']},
    'pigeonpeas':  {'emoji':'🫛','latin':'Cajanus cajan',         'season':'Kharif · Jun–Nov',    'price':'₹55–80 / kg',  'priceKg':67,  'yieldStr':'1–2 t/ha',    'yieldMid':1.5,  'water':'Low',         'tips':['Intercrop with cereals to maximise land use','Pinch main shoot at 15 cm to encourage branching','Apply phosphorus as a basal dose at sowing']},
    'mothbeans':   {'emoji':'🫘','latin':'Vigna aconitifolia',    'season':'Kharif · Jun–Sep',    'price':'₹40–60 / kg',  'priceKg':50,  'yieldStr':'0.4–0.8 t/ha','yieldMid':0.6,  'water':'Very Low',    'tips':['Extremely drought-tolerant — ideal for arid regions','Shallow ploughing conserves soil moisture','No supplemental irrigation needed in most situations']},
    'mungbean':    {'emoji':'🫘','latin':'Vigna radiata',         'season':'Zaid / Kharif',       'price':'₹70–90 / kg',  'priceKg':80,  'yieldStr':'0.6–1.2 t/ha','yieldMid':0.9,  'water':'Low',         'tips':['Short-duration crop ready in 60–75 days','Excellent green-manure to enrich soil nitrogen','Avoid standing water; waterlogging causes root rot']},
    'blackgram':   {'emoji':'⚫','latin':'Vigna mungo',            'season':'Kharif / Rabi',       'price':'₹60–80 / kg',  'priceKg':70,  'yieldStr':'0.5–1.0 t/ha','yieldMid':0.7,  'water':'Low',         'tips':['Grows well on residual soil moisture after kharif','Apply Rhizobium seed treatment for nitrogen fixation','Harvest when 70% of pods have turned black']},
    'lentil':      {'emoji':'🫘','latin':'Lens culinaris',        'season':'Rabi · Oct–Mar',      'price':'₹80–110 / kg', 'priceKg':95,  'yieldStr':'0.5–1.0 t/ha','yieldMid':0.75, 'water':'Low',         'tips':['Sow in rows 25–30 cm apart for good air circulation','Cool temperatures 18–25°C during pod filling maximise yield','Avoid excessive irrigation; waterlogging causes root rot']},
    'cotton':      {'emoji':'🌿','latin':'Gossypium hirsutum',    'season':'Kharif · May–Nov',    'price':'₹55–65 / kg',  'priceKg':60,  'yieldStr':'1.5–2.5 t/ha','yieldMid':2.0,  'water':'High',        'tips':['Well-drained black cotton soil gives the best fibre quality','Apply boron at bud initiation to improve boll set','Intercrop with cowpea to suppress weeds naturally']},
    'jute':        {'emoji':'🌿','latin':'Corchorus olitorius',   'season':'Kharif · Mar–Jul',    'price':'₹35–50 / kg',  'priceKg':42,  'yieldStr':'2–3.5 t/ha',  'yieldMid':2.75, 'water':'High',        'tips':['Retting in slow-moving water produces finest fibre','Weed control in the first six weeks is critical','Harvest at 50% flowering for finest fibre grade']},
    'coffee':      {'emoji':'☕','latin':'Coffea arabica',        'season':'Perennial · Oct–Jan', 'price':'₹350–500 / kg','priceKg':420, 'yieldStr':'0.5–1.5 t/ha','yieldMid':1.0,  'water':'Medium',      'tips':['Grow under 30–40% shade for premium cup quality','Well-drained laterite soil rich in organic matter is ideal','Regular pruning improves air circulation and yield']},
    'banana':      {'emoji':'🍌','latin':'Musa acuminata',        'season':'Year-round',          'price':'₹12–20 / kg',  'priceKg':16,  'yieldStr':'25–40 t/ha',  'yieldMid':32,   'water':'High',        'tips':['Plant at 1.8x1.5 m spacing for optimal bunch size','Apply potassium in 3–4 split doses during growth','Remove all but one healthy follower sucker']},
    'mango':       {'emoji':'🥭','latin':'Mangifera indica',      'season':'Summer · Apr–Jun',    'price':'₹30–80 / kg',  'priceKg':50,  'yieldStr':'8–12 t/ha',   'yieldMid':10,   'water':'Low',         'tips':['Requires a dry cool period to initiate flowering','Apply potassium and boron before flowering season','Regular pruning after harvest improves light penetration']},
    'grapes':      {'emoji':'🍇','latin':'Vitis vinifera',        'season':'Harvest · Nov–Mar',   'price':'₹40–120 / kg', 'priceKg':70,  'yieldStr':'8–15 t/ha',   'yieldMid':11.5, 'water':'Medium',      'tips':['Very high K requirement for sugar accumulation','Prune to 2-bud spurs in January for better fruit set','Control powdery mildew early with sulphur-based sprays']},
    'watermelon':  {'emoji':'🍉','latin':'Citrullus lanatus',     'season':'Zaid · Feb–May',      'price':'₹6–12 / kg',   'priceKg':9,   'yieldStr':'25–40 t/ha',  'yieldMid':32,   'water':'Medium',      'tips':['Sandy loam soil with excellent drainage is ideal','Apply mulch to conserve soil moisture and suppress weeds','Avoid insecticides during flowering — bees are essential']},
    'muskmelon':   {'emoji':'🍈','latin':'Cucumis melo',          'season':'Zaid · Feb–May',      'price':'₹10–20 / kg',  'priceKg':15,  'yieldStr':'15–20 t/ha',  'yieldMid':17.5, 'water':'Medium',      'tips':['Warm days and cool nights are essential for high sugar content','Reduce irrigation 2 weeks before harvest for sweetness','Use netting under fruits to prevent soil contact and rot']},
    'apple':       {'emoji':'🍎','latin':'Malus domestica',       'season':'Autumn · Aug–Oct',    'price':'₹60–150 / kg', 'priceKg':100, 'yieldStr':'10–20 t/ha',  'yieldMid':15,   'water':'Medium',      'tips':['Requires 1,000+ chilling hours below 7°C for fruiting','Apply high P and K in pre-blossom stage','Thin to one fruit per spur to improve fruit size']},
    'orange':      {'emoji':'🍊','latin':'Citrus sinensis',       'season':'Winter · Nov–Feb',    'price':'₹20–40 / kg',  'priceKg':30,  'yieldStr':'10–20 t/ha',  'yieldMid':15,   'water':'Medium',      'tips':['Deep sandy loam with excellent drainage is most suitable','Apply zinc and iron micronutrients to prevent chlorosis','Irrigate at blossom drop stage to improve fruit set']},
    'papaya':      {'emoji':'🧡','latin':'Carica papaya',         'season':'Year-round',          'price':'₹10–20 / kg',  'priceKg':15,  'yieldStr':'35–50 t/ha',  'yieldMid':42,   'water':'Medium',      'tips':['Plant on raised beds to prevent crown rot','Maintain 1 male plant per 10 female plants','First harvest possible within 9–11 months of planting']},
    'coconut':     {'emoji':'🥥','latin':'Cocos nucifera',        'season':'Year-round',          'price':'₹20–40 / nut', 'priceKg':30,  'yieldStr':'10–15 t/ha',  'yieldMid':12,   'water':'High',        'tips':['Coastal sandy loam with high humidity is ideal','Apply NPK + micronutrients every 6 months','Regularly remove old fronds to prevent pest harbourage']},
    'pomegranate': {'emoji':'🍑','latin':'Punica granatum',       'season':'Feb–Mar & Jun–Aug',   'price':'₹50–100 / kg', 'priceKg':70,  'yieldStr':'8–12 t/ha',   'yieldMid':10,   'water':'Low',         'tips':['Once established, highly drought-tolerant','Thin to 2–3 fruits per branch for superior fruit size','Spray boron and zinc at bud initiation for better fruit set']},
}

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgriSense – Crop Recommendation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,600&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --ink:#0D1208;--ink2:#1A2315;--ink3:#243020;
  --gold:#C9932A;--gold2:#E3B55A;--gold3:#F0D08A;
  --sage:#4A6741;--sage2:#6B9462;
  --cream:#F7F2E8;--rust2:#C8573F;--white:#FDFCF8;
  --r:14px;--r2:20px;--r3:28px;
}
html{font-size:15px;scroll-behavior:smooth}
body{font-family:'Outfit',sans-serif;background:var(--ink);color:var(--cream);min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='1'/%3E%3C/svg%3E");opacity:.032;pointer-events:none;z-index:9999}

/* HEADER */
.header{position:sticky;top:0;z-index:100;background:rgba(13,18,8,.93);backdrop-filter:blur(18px) saturate(160%);border-bottom:1px solid rgba(201,147,42,.18);padding:0 40px;height:68px;display:flex;align-items:center;gap:16px}
.logo-mark{width:40px;height:40px;border:1.5px solid var(--gold);border-radius:10px;display:flex;align-items:center;justify-content:center;position:relative;flex-shrink:0}
.logo-mark::after{content:'';position:absolute;inset:3px;border-radius:6px;background:rgba(201,147,42,.1)}
.logo-mark svg{width:20px;height:20px;fill:var(--gold);position:relative;z-index:1}
.brand{font-family:'Cormorant Garamond',serif;font-size:24px;font-weight:600;color:var(--cream);letter-spacing:.5px}
.brand-sub{font-size:10px;color:rgba(201,147,42,.65);letter-spacing:2.5px;text-transform:uppercase;margin-top:-2px}
.header-pill{margin-left:auto;background:rgba(201,147,42,.1);border:1px solid rgba(201,147,42,.28);color:var(--gold2);font-size:11px;font-weight:500;padding:5px 14px;border-radius:20px;letter-spacing:.5px;display:flex;align-items:center;gap:7px}
.header-pill::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--gold2);box-shadow:0 0 6px var(--gold2);animation:pdot 2s ease-in-out infinite}
@keyframes pdot{0%,100%{opacity:1}50%{opacity:.25}}
.acc-badge{font-size:11px;background:rgba(74,103,65,.3);border:1px solid rgba(107,148,98,.4);color:var(--sage2);padding:4px 10px;border-radius:20px;display:none}

/* HERO */
.hero-strip{background:linear-gradient(135deg,var(--ink2) 0%,var(--ink3) 60%,#1E2C19 100%);border-bottom:1px solid rgba(201,147,42,.1);padding:40px 40px 38px;position:relative;overflow:hidden}
.hero-strip::before{content:'';position:absolute;top:-60px;right:-60px;width:300px;height:300px;border-radius:50%;background:radial-gradient(circle,rgba(201,147,42,.07) 0%,transparent 70%);pointer-events:none}
.hero-eyebrow{font-size:11px;font-weight:500;color:var(--gold);letter-spacing:2.5px;text-transform:uppercase;margin-bottom:10px}
.hero-title{font-family:'Cormorant Garamond',serif;font-size:clamp(28px,4vw,42px);font-weight:600;line-height:1.15;color:var(--white);margin-bottom:10px}
.hero-title em{font-style:italic;color:var(--gold2)}
.hero-desc{font-size:13.5px;color:rgba(215,205,188,.55);max-width:560px;line-height:1.7}
.model-info{display:flex;align-items:center;gap:8px;margin-top:16px;font-size:12px;color:rgba(215,205,188,.35)}
.model-info span{background:rgba(255,255,255,.04);border:1px solid rgba(215,205,188,.1);padding:4px 10px;border-radius:6px}

/* LAYOUT */
.main-wrap{display:grid;grid-template-columns:420px 1fr;min-height:calc(100vh - 68px - 106px)}

/* LEFT */
.panel-left{background:var(--ink2);border-right:1px solid rgba(201,147,42,.1);padding:32px;position:sticky;top:68px;height:calc(100vh - 68px);overflow-y:auto}
.panel-left::-webkit-scrollbar{width:3px}
.panel-left::-webkit-scrollbar-thumb{background:rgba(201,147,42,.25);border-radius:2px}
.sec-label{font-size:10px;font-weight:600;color:var(--gold);letter-spacing:2px;text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.sec-label::after{content:'';flex:1;height:1px;background:linear-gradient(to right,rgba(201,147,42,.3),transparent)}
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:24px}
.fg{display:flex;flex-direction:column;gap:5px}
.fg .lbl{font-size:10px;font-weight:600;color:rgba(215,205,188,.4);letter-spacing:1.5px;text-transform:uppercase}
.fg .unit{font-size:10px;color:rgba(201,147,42,.45);letter-spacing:.5px}
.fg input{background:rgba(255,255,255,.04);border:1px solid rgba(215,205,188,.11);border-radius:10px;padding:10px 14px;font-size:14px;font-family:'Outfit',sans-serif;color:var(--cream);transition:all .2s;outline:none}
.fg input::placeholder{color:rgba(215,205,188,.18)}
.fg input:focus{border-color:rgba(201,147,42,.5);background:rgba(201,147,42,.06);box-shadow:0 0 0 3px rgba(201,147,42,.08)}
.fg input:hover:not(:focus){border-color:rgba(215,205,188,.2)}
.ph-block{margin-bottom:24px}
.ph-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px}
.ph-number{font-family:'Cormorant Garamond',serif;font-size:30px;font-weight:600;color:var(--gold2);line-height:1}
.ph-subdesc{font-size:11px;color:rgba(215,205,188,.35);margin-top:2px}
.ph-scale-hint{text-align:right;font-size:10px;color:rgba(215,205,188,.25);line-height:1.9}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:4px;border-radius:2px;background:linear-gradient(to right,var(--rust2) 0%,var(--gold) 38%,var(--sage2) 65%,var(--sage) 100%);outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:20px;height:20px;border-radius:50%;background:var(--gold2);border:2.5px solid var(--ink);box-shadow:0 2px 10px rgba(201,147,42,.4);transition:transform .15s,box-shadow .15s;cursor:pointer}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.15);box-shadow:0 2px 16px rgba(201,147,42,.65)}
.ph-markers{display:flex;justify-content:space-between;margin-top:6px}
.ph-markers span{font-size:10px;color:rgba(215,205,188,.25)}
.hairline{border:none;border-top:1px solid rgba(215,205,188,.07);margin:20px 0 24px}
.area-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:26px}
.fg select{background:rgba(255,255,255,.04);border:1px solid rgba(215,205,188,.11);border-radius:10px;padding:10px 14px;font-size:14px;font-family:'Outfit',sans-serif;color:var(--cream);outline:none;cursor:pointer;transition:border-color .2s}
.fg select:focus{border-color:rgba(201,147,42,.5)}
.fg select option{background:var(--ink2)}
.btn-cta{width:100%;padding:15px 20px;background:linear-gradient(135deg,var(--gold) 0%,var(--gold2) 100%);border:none;border-radius:12px;font-family:'Outfit',sans-serif;font-size:15px;font-weight:600;color:var(--ink);cursor:pointer;display:flex;align-items:center;justify-content:center;gap:10px;transition:all .25s;letter-spacing:.3px;box-shadow:0 4px 20px rgba(201,147,42,.3);position:relative;overflow:hidden}
.btn-cta::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(255,255,255,.15) 0%,transparent 60%);pointer-events:none}
.btn-cta:hover{transform:translateY(-2px);box-shadow:0 8px 32px rgba(201,147,42,.48)}
.btn-cta:active{transform:translateY(0)}
.btn-cta:disabled{opacity:.5;cursor:not-allowed;transform:none}
.btn-cta svg{width:18px;height:18px}

/* RIGHT */
.panel-right{background:var(--ink);padding:32px;min-height:100%}
.placeholder{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:500px;gap:20px;text-align:center}
.ph-icon-wrap{width:100px;height:100px;border-radius:50%;border:1px dashed rgba(201,147,42,.22);display:flex;align-items:center;justify-content:center;position:relative}
.ph-icon-wrap::before{content:'';position:absolute;inset:-8px;border-radius:50%;border:1px dashed rgba(201,147,42,.08)}
.ph-icon-wrap svg{width:44px;height:44px;fill:rgba(201,147,42,.22)}
.placeholder h3{font-family:'Cormorant Garamond',serif;font-size:22px;font-weight:500;color:rgba(215,205,188,.35)}
.placeholder p{font-size:13px;color:rgba(215,205,188,.22);max-width:280px;line-height:1.75}
.loading{display:none;flex-direction:column;align-items:center;justify-content:center;min-height:500px;gap:18px}
.loader-ring{width:52px;height:52px;border-radius:50%;border:2px solid rgba(201,147,42,.12);border-top-color:var(--gold);animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.loading span{font-size:11px;color:rgba(201,147,42,.5);letter-spacing:2px;text-transform:uppercase}

/* ERROR STATE */
.error-box{display:none;background:rgba(168,61,47,.1);border:1px solid rgba(168,61,47,.35);border-radius:var(--r2);padding:22px;margin-bottom:14px;color:rgba(240,180,160,.8);font-size:13px;line-height:1.65}
.error-box strong{display:block;font-size:15px;margin-bottom:6px;color:#F09070}

.result{display:none;animation:fadeUp .5s ease both}
@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}

/* CROP HERO */
.crop-hero{background:linear-gradient(135deg,var(--ink2) 0%,var(--ink3) 100%);border:1px solid rgba(201,147,42,.2);border-radius:var(--r3);padding:28px 30px;margin-bottom:14px;position:relative;overflow:hidden}
.crop-hero::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(to right,transparent,rgba(201,147,42,.5),transparent)}
.crop-hero::after{content:'';position:absolute;bottom:-80px;right:-80px;width:220px;height:220px;border-radius:50%;background:radial-gradient(circle,rgba(201,147,42,.06) 0%,transparent 70%)}
.ch-row1{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:8px}
.ch-emoji{font-size:46px;line-height:1}
.best-badge{background:rgba(201,147,42,.14);border:1px solid rgba(201,147,42,.38);color:var(--gold2);font-size:11px;font-weight:600;padding:5px 12px;border-radius:20px;letter-spacing:.8px;text-transform:uppercase}
.ch-name{font-family:'Cormorant Garamond',serif;font-size:46px;font-weight:700;line-height:1.05;color:var(--white);text-transform:capitalize;letter-spacing:-.5px}
.ch-latin{font-family:'Cormorant Garamond',serif;font-size:16px;font-style:italic;color:rgba(201,147,42,.55);margin-top:3px}
.conf-bar{margin-top:22px}
.conf-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}
.conf-label{font-size:10px;font-weight:600;color:rgba(215,205,188,.3);letter-spacing:1.5px;text-transform:uppercase}
.conf-pct{font-family:'Cormorant Garamond',serif;font-size:28px;font-weight:600;color:var(--gold2)}
.conf-track{height:3px;background:rgba(255,255,255,.07);border-radius:2px;overflow:hidden}
.conf-fill{height:100%;width:0;background:linear-gradient(to right,var(--gold),var(--gold2));border-radius:2px;transition:width 1.2s cubic-bezier(.4,0,.2,1);box-shadow:0 0 8px rgba(201,147,42,.5)}

/* CHIPS */
.chips-row{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:14px}
.chip{background:rgba(255,255,255,.04);border:1px solid rgba(215,205,188,.09);border-radius:20px;padding:5px 12px;font-size:12px;color:rgba(215,205,188,.5);display:flex;align-items:center;gap:5px}
.chip strong{color:var(--cream);font-weight:500}

/* STATS */
.stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px}
.stat-box{background:var(--ink2);border:1px solid rgba(201,147,42,.09);border-radius:var(--r);padding:14px 16px;transition:border-color .2s}
.stat-box:hover{border-color:rgba(201,147,42,.25)}
.sb-label{font-size:10px;font-weight:600;color:rgba(215,205,188,.28);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px}
.sb-val{font-size:14px;font-weight:500;color:var(--cream);line-height:1.3}
.sb-val.gold{color:var(--gold2)}

/* REVENUE */
.rev-panel{background:var(--ink2);border:1px solid rgba(201,147,42,.15);border-radius:var(--r2);padding:22px;margin-bottom:14px;position:relative;overflow:hidden}
.rev-panel::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(to right,transparent,rgba(201,147,42,.35),transparent)}
.rev-head{font-size:10px;font-weight:600;color:var(--gold);letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;display:flex;align-items:center;gap:8px}
.rev-head svg{width:14px;height:14px}
.rev-cards{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.rev-card{background:rgba(201,147,42,.06);border:1px solid rgba(201,147,42,.14);border-radius:var(--r);padding:18px;text-align:center}
.rc-label{font-size:10px;font-weight:600;color:rgba(201,147,42,.55);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px}
.rc-val{font-family:'Cormorant Garamond',serif;font-size:30px;font-weight:600;color:var(--gold2);line-height:1}
.rc-sub{font-size:11px;color:rgba(215,205,188,.28);margin-top:5px}

/* ALTERNATIVES */
.alt-panel{background:var(--ink2);border:1px solid rgba(215,205,188,.07);border-radius:var(--r2);padding:22px;margin-bottom:14px}
.panel-head{font-size:10px;font-weight:600;color:rgba(215,205,188,.35);letter-spacing:2px;text-transform:uppercase;margin-bottom:16px}
.alt-item{display:flex;align-items:center;gap:12px;padding:9px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.alt-item:last-child{border-bottom:none;padding-bottom:0}
.alt-num{font-family:'Cormorant Garamond',serif;font-size:18px;font-weight:500;color:rgba(201,147,42,.28);width:22px;flex-shrink:0}
.alt-emoji{font-size:18px;flex-shrink:0}
.alt-name{font-size:14px;font-weight:500;color:var(--cream);text-transform:capitalize;min-width:120px}
.alt-bar-bg{flex:1;height:3px;background:rgba(255,255,255,.05);border-radius:2px;overflow:hidden}
.alt-bar-fill{height:100%;background:linear-gradient(to right,rgba(201,147,42,.35),rgba(201,147,42,.65));border-radius:2px;transition:width .9s ease}
.alt-pct{font-size:12px;color:rgba(201,147,42,.5);min-width:36px;text-align:right;font-weight:500}

/* TIPS */
.tips-panel{background:var(--ink2);border:1px solid rgba(215,205,188,.07);border-radius:var(--r2);padding:22px}
.tip-item{display:flex;gap:14px;padding:11px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.tip-item:first-child{padding-top:0}
.tip-item:last-child{border-bottom:none;padding-bottom:0}
.tip-num{font-family:'Cormorant Garamond',serif;font-size:22px;font-weight:600;line-height:1;color:rgba(201,147,42,.22);flex-shrink:0;width:20px}
.tip-text{font-size:13px;color:rgba(215,205,188,.58);line-height:1.7;padding-top:2px}

/* FOOTER */
.footer{background:var(--ink2);border-top:1px solid rgba(201,147,42,.1);padding:16px 40px;display:flex;align-items:center;justify-content:space-between;font-size:11px;color:rgba(215,205,188,.22)}
.footer strong{color:rgba(201,147,42,.45)}

/* STAGGER */
.result>*{animation:fadeUp .4s ease both}
.result>*:nth-child(1){animation-delay:.05s}
.result>*:nth-child(2){animation-delay:.1s}
.result>*:nth-child(3){animation-delay:.15s}
.result>*:nth-child(4){animation-delay:.2s}
.result>*:nth-child(5){animation-delay:.25s}
.result>*:nth-child(6){animation-delay:.3s}
.result>*:nth-child(7){animation-delay:.35s}

/* RESPONSIVE */
@media(max-width:900px){.main-wrap{grid-template-columns:1fr}.panel-left{position:static;height:auto;border-right:none;border-bottom:1px solid rgba(201,147,42,.1)}.stats-grid{grid-template-columns:1fr 1fr}.hero-strip,.header,.panel-left,.panel-right,.footer{padding-left:20px;padding-right:20px}}
@media(max-width:500px){.form-grid,.area-row,.rev-cards{grid-template-columns:1fr}.stats-grid{grid-template-columns:1fr 1fr}.header-pill{display:none}}
</style>
</head>
<body>

<header class="header">
  <div class="logo-mark">
    <svg viewBox="0 0 24 24"><path d="M17 8C8 10 5.9 16.17 3.82 19.61L5.71 21l1-2.3A4.49 4.49 0 008 19c9-3 6.5-8 6.5-8s-.18 5.08-4.5 6.5c-.64.22-1.33.5-1.89.85L10 21.5c2.77-2.5 6.3-5.5 9-4.5 0 0-2 6-11 7 3 0 12-1 14.5-13.5C23 8 17 8 17 8z"/></svg>
  </div>
  <div>
    <div class="brand">AgriSense</div>
    <div class="brand-sub">Precision Farming</div>
  </div>
  <div class="header-pill">Live RF Model</div>
  <div class="acc-badge" id="acc-badge">RF Accuracy: —</div>
</header>

<div class="hero-strip">
  <div class="hero-eyebrow">Real Machine Learning Backend</div>
  <h1 class="hero-title">Discover the <em>right crop</em><br>for your land</h1>
  <p class="hero-desc">Enter your soil and climate conditions. Your inputs are sent to a live Python Flask server running a trained <strong style="color:var(--gold3)">Random Forest Classifier</strong> — the exact model from <code style="font-size:12px;color:var(--sage3);background:rgba(255,255,255,.05);padding:2px 6px;border-radius:4px">itaprojectcode.py</code>.</p>
  <div class="model-info">
    <span>RandomForestClassifier</span>
    <span>n_estimators = 100</span>
    <span>random_state = 42</span>
    <span>2,200 training samples</span>
    <span>22 crops</span>
  </div>
</div>

<div class="main-wrap">

  <aside class="panel-left">

    <div class="sec-label">Soil Nutrients</div>
    <div class="form-grid">
      <div class="fg"><span class="lbl">Nitrogen (N)</span><input type="number" id="inp-N" placeholder="e.g. 90" min="0" max="200"><span class="unit">kg / ha</span></div>
      <div class="fg"><span class="lbl">Phosphorus (P)</span><input type="number" id="inp-P" placeholder="e.g. 42" min="0" max="200"><span class="unit">kg / ha</span></div>
      <div class="fg"><span class="lbl">Potassium (K)</span><input type="number" id="inp-K" placeholder="e.g. 43" min="0" max="300"><span class="unit">kg / ha</span></div>
    </div>

    <div class="sec-label">Climate</div>
    <div class="form-grid">
      <div class="fg"><span class="lbl">Temperature</span><input type="number" id="inp-temp" placeholder="e.g. 25" min="0" max="55" step="0.1"><span class="unit">°C</span></div>
      <div class="fg"><span class="lbl">Humidity</span><input type="number" id="inp-hum" placeholder="e.g. 70" min="0" max="100" step="0.1"><span class="unit">%</span></div>
      <div class="fg"><span class="lbl">Rainfall</span><input type="number" id="inp-rain" placeholder="e.g. 200" min="0" max="500" step="0.1"><span class="unit">mm</span></div>
    </div>

    <div class="sec-label">Soil pH</div>
    <div class="ph-block">
      <div class="ph-top">
        <div>
          <div class="ph-number" id="ph-disp">6.5</div>
          <div class="ph-subdesc" id="ph-desc">Slightly acidic</div>
        </div>
        <div class="ph-scale-hint"><div>3 – Highly acidic</div><div>7 – Neutral</div><div>9.9 – Alkaline</div></div>
      </div>
      <input type="range" id="ph-slider" min="3" max="9.9" step="0.1" value="6.5" oninput="updatePH(this.value)">
      <div class="ph-markers"><span>Acidic</span><span>Neutral</span><span>Alkaline</span></div>
    </div>

    <hr class="hairline">

    <div class="sec-label">Farm Area <span style="font-size:9px;color:rgba(215,205,188,.22);text-transform:none;letter-spacing:0;font-weight:400">&nbsp;optional</span></div>
    <div class="area-row">
      <div class="fg"><span class="lbl">Size</span><input type="number" id="inp-area" placeholder="e.g. 2.5" min="0" step="0.01"></div>
      <div class="fg"><span class="lbl">Unit</span>
        <select id="inp-unit">
          <option value="hectares">Hectares (ha)</option>
          <option value="acres">Acres</option>
          <option value="sqm">Square Metres</option>
          <option value="bigha">Bigha</option>
        </select>
      </div>
    </div>

    <button class="btn-cta" id="btn-submit" onclick="runModel()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
      Analyse &amp; Recommend
    </button>

  </aside>

  <main class="panel-right">

    <div class="placeholder" id="el-ph">
      <div class="ph-icon-wrap">
        <svg viewBox="0 0 24 24"><path d="M17 8C8 10 5.9 16.17 3.82 19.61L5.71 21l1-2.3A4.49 4.49 0 008 19c9-3 6.5-8 6.5-8s-.18 5.08-4.5 6.5c-.64.22-1.33.5-1.89.85L10 21.5c2.77-2.5 6.3-5.5 9-4.5 0 0-2 6-11 7 3 0 12-1 14.5-13.5C23 8 17 8 17 8z"/></svg>
      </div>
      <h3>Awaiting your parameters</h3>
      <p>Fill in your soil and climate data on the left, then click <strong style="color:var(--gold2)">Analyse &amp; Recommend</strong>.</p>
    </div>

    <div class="loading" id="el-ldg">
      <div class="loader-ring"></div>
      <span>Calling Flask server</span>
    </div>

    <div class="error-box" id="el-err">
      <strong>Could not reach the Flask server</strong>
      Make sure you have started the backend with <code style="background:rgba(255,255,255,.08);padding:2px 6px;border-radius:4px">python3 app.py</code> in your terminal, then try again.
    </div>

    <div class="result" id="el-res">

      <div class="crop-hero">
        <div class="ch-row1">
          <span class="ch-emoji" id="r-emoji">🌾</span>
          <span class="best-badge">Best Match · RF Model</span>
        </div>
        <div class="ch-name" id="r-name">—</div>
        <div class="ch-latin" id="r-latin">—</div>
        <div class="conf-bar">
          <div class="conf-top">
            <span class="conf-label">RF Prediction Confidence</span>
            <span class="conf-pct" id="r-conf">—</span>
          </div>
          <div class="conf-track"><div class="conf-fill" id="r-conf-fill"></div></div>
        </div>
      </div>

      <div class="chips-row" id="r-chips"></div>

      <div class="stats-grid">
        <div class="stat-box"><div class="sb-label">Season</div><div class="sb-val" id="r-season">—</div></div>
        <div class="stat-box"><div class="sb-label">Market Price</div><div class="sb-val gold" id="r-price">—</div></div>
        <div class="stat-box"><div class="sb-label">Avg. Yield</div><div class="sb-val" id="r-yield">—</div></div>
        <div class="stat-box"><div class="sb-label">Water Need</div><div class="sb-val" id="r-water">—</div></div>
      </div>

      <div class="rev-panel" id="el-rev">
        <div class="rev-head">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6"/></svg>
          Revenue Estimate
        </div>
        <div class="rev-cards">
          <div class="rev-card"><div class="rc-label">Est. Harvest</div><div class="rc-val" id="r-harvest">—</div><div class="rc-sub">tonnes</div></div>
          <div class="rev-card"><div class="rc-label">Est. Revenue</div><div class="rc-val" id="r-revenue">—</div><div class="rc-sub">at mid-market rate</div></div>
        </div>
      </div>

      <div class="alt-panel">
        <div class="panel-head">Other viable crops (RF probabilities)</div>
        <div id="r-alts"></div>
      </div>

      <div class="tips-panel">
        <div class="panel-head">Growing Tips</div>
        <div id="r-tips"></div>
      </div>

    </div>
  </main>
</div>

<footer class="footer">
  <span>AgriSense &mdash; <strong>Crop_recommendation.csv</strong> &middot; 2,200 rows &middot; 22 crops &middot; 7 features</span>
  <span>Backend: <strong>Flask + RandomForestClassifier</strong> (n_estimators=100)</span>
</footer>

<script>
const EMOJI = {rice:'🌾',maize:'🌽',chickpea:'🫘',kidneybeans:'🫘',pigeonpeas:'🫛',mothbeans:'🫘',mungbean:'🫘',blackgram:'⚫',lentil:'🫘',cotton:'🌿',jute:'🌿',coffee:'☕',banana:'🍌',mango:'🥭',grapes:'🍇',watermelon:'🍉',muskmelon:'🍈',apple:'🍎',orange:'🍊',papaya:'🧡',coconut:'🥥',pomegranate:'🍑'};

function updatePH(v){
  const n=parseFloat(v).toFixed(1);
  document.getElementById('ph-disp').textContent=n;
  const d=parseFloat(n);
  document.getElementById('ph-desc').textContent=d<4.5?'Extremely acidic':d<5.5?'Strongly acidic':d<6.0?'Moderately acidic':d<6.5?'Slightly acidic':d<7.0?'Near neutral':d<7.5?'Neutral':d<8.0?'Slightly alkaline':'Strongly alkaline';
}

async function runModel(){
  const N=parseFloat(document.getElementById('inp-N').value);
  const P=parseFloat(document.getElementById('inp-P').value);
  const K=parseFloat(document.getElementById('inp-K').value);
  const temp=parseFloat(document.getElementById('inp-temp').value);
  const hum=parseFloat(document.getElementById('inp-hum').value);
  const rain=parseFloat(document.getElementById('inp-rain').value);
  const ph=parseFloat(document.getElementById('ph-slider').value);
  const area=parseFloat(document.getElementById('inp-area').value)||0;
  const unit=document.getElementById('inp-unit').value;

  const miss=[];
  if(isNaN(N))miss.push('Nitrogen');
  if(isNaN(P))miss.push('Phosphorus');
  if(isNaN(K))miss.push('Potassium');
  if(isNaN(temp))miss.push('Temperature');
  if(isNaN(hum))miss.push('Humidity');
  if(isNaN(rain))miss.push('Rainfall');
  if(miss.length){alert('Please fill in: '+miss.join(', '));return}

  // Show loading
  document.getElementById('el-ph').style.display='none';
  document.getElementById('el-res').style.display='none';
  document.getElementById('el-err').style.display='none';
  document.getElementById('el-ldg').style.display='flex';
  document.getElementById('btn-submit').disabled=true;
  document.getElementById('r-conf-fill').style.width='0%';

  try {
    // ── Call Flask backend ──
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({N, P, K, temperature:temp, humidity:hum, ph, rainfall:rain, area, unit})
    });

    if (!response.ok) throw new Error(`Server error: ${response.status}`);
    const data = await response.json();
    if (data.error) throw new Error(data.error);

    document.getElementById('el-ldg').style.display='none';
    document.getElementById('btn-submit').disabled=false;

    // Show accuracy badge
    const badge = document.getElementById('acc-badge');
    badge.textContent = `RF Accuracy: ${data.model_accuracy}%`;
    badge.style.display = 'flex';

    // Hero
    document.getElementById('r-emoji').textContent = data.emoji;
    document.getElementById('r-name').textContent  = data.crop.charAt(0).toUpperCase()+data.crop.slice(1);
    document.getElementById('r-latin').textContent = data.latin;
    document.getElementById('r-conf').textContent  = data.confidence+'%';
    setTimeout(()=>{document.getElementById('r-conf-fill').style.width=data.confidence+'%'},100);

    // Input chips
    document.getElementById('r-chips').innerHTML=[
      ['N',N,'kg/ha'],['P',P,'kg/ha'],['K',K,'kg/ha'],
      ['Temp',temp+'°C',''],['Humidity',hum+'%',''],
      ['pH',parseFloat(ph).toFixed(1),''],['Rain',rain+' mm','']
    ].map(([l,v,u])=>`<div class="chip"><strong>${l}</strong>${v}${u}</div>`).join('');

    // Stats
    document.getElementById('r-season').textContent = data.season;
    document.getElementById('r-price').textContent  = data.price;
    document.getElementById('r-yield').textContent  = data.yieldStr;
    document.getElementById('r-water').textContent  = data.water;

    // Revenue
    const revEl=document.getElementById('el-rev');
    if(data.revenue){
      document.getElementById('r-harvest').textContent = data.revenue.harvest_tonnes;
      document.getElementById('r-revenue').textContent = '₹'+data.revenue.revenue_inr.toLocaleString('en-IN');
      revEl.style.display='block';
    }else{revEl.style.display='none'}

    // Alternatives
    const topConf = data.confidence;
    document.getElementById('r-alts').innerHTML = data.alternatives.map((a,i)=>{
      const pct=Math.round(a.confidence/topConf*100);
      const em=EMOJI[a.crop]||'🌱';
      return`<div class="alt-item"><span class="alt-num">${i+2}</span><span class="alt-emoji">${em}</span><span class="alt-name">${a.crop.charAt(0).toUpperCase()+a.crop.slice(1)}</span><div class="alt-bar-bg"><div class="alt-bar-fill" style="width:${pct}%"></div></div><span class="alt-pct">${a.confidence}%</span></div>`;
    }).join('');

    // Tips
    document.getElementById('r-tips').innerHTML = data.tips.map((t,i)=>
      `<div class="tip-item"><span class="tip-num">0${i+1}</span><p class="tip-text">${t}</p></div>`
    ).join('');

    document.getElementById('el-res').style.display='block';
    document.getElementById('el-res').scrollIntoView({behavior:'smooth',block:'start'});

  } catch(err) {
    document.getElementById('el-ldg').style.display='none';
    document.getElementById('btn-submit').disabled=false;
    document.getElementById('el-err').style.display='block';
    console.error('AgriSense API error:', err);
  }
}

document.addEventListener('keydown',e=>{if(e.key==='Enter')runModel()});
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_PAGE

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([{
            'N':           float(data['N']),
            'P':           float(data['P']),
            'K':           float(data['K']),
            'temperature': float(data['temperature']),
            'humidity':    float(data['humidity']),
            'ph':          float(data['ph']),
            'rainfall':    float(data['rainfall']),
        }])
        proba   = rf_model.predict_proba(input_df)[0]
        classes = rf_model.classes_
        ranked  = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)

        best_crop = ranked[0][0]
        best_conf = round(ranked[0][1] * 100, 1)
        alts      = [{'crop': k, 'confidence': round(v * 100, 1)} for k, v in ranked[1:5]]
        meta      = CROP_META.get(best_crop, {})

        revenue_data = None
        area = float(data.get('area', 0))
        if area > 0:
            unit = data.get('unit', 'hectares')
            ha = area
            if unit == 'acres':  ha = area * 0.404686
            if unit == 'sqm':    ha = area / 10000
            if unit == 'bigha':  ha = area * 0.2
            harvest_t   = round(meta.get('yieldMid', 2) * ha, 2)
            revenue_inr = round(harvest_t * 1000 * meta.get('priceKg', 20))
            revenue_data = {'harvest_tonnes': harvest_t, 'revenue_inr': revenue_inr}

        return jsonify({
            'success': True, 'crop': best_crop, 'confidence': best_conf,
            'emoji': meta.get('emoji','🌱'), 'latin': meta.get('latin',''),
            'season': meta.get('season','—'), 'price': meta.get('price','—'),
            'yieldStr': meta.get('yieldStr','—'), 'water': meta.get('water','—'),
            'tips': meta.get('tips',[]), 'alternatives': alts,
            'revenue': revenue_data, 'model_accuracy': round(accuracy, 2),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status':'ok','accuracy':round(accuracy,2),'crops':len(rf_model.classes_)})

if __name__ == '__main__':
    print("\n AgriSense running at http://localhost:5000\n")
    app.run(debug=False, port=5000)