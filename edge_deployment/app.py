"""
CARE Wind Turbine Predictive Maintenance & Anomaly Detection Dashboard.
Runs on Raspberry Pi (ARM CPU) or Laptop.
Real-Time SCADA Simulation, Physics Residual Monitoring, and Zero-Shot CARE Evaluation.
"""

import os
import sys
import time
import json
import psutil
from flask import Flask, render_template, request, jsonify, Response
import pandas as pd

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model_core import ModelInferenceEngine
from data_manager import LiveSimulationManager, DatasetReplayManager

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

# Locate weights
weights_candidates = [
    os.path.join(BASE_DIR, 'weights', 'model_epoch17.ckpt'),
    os.path.join(BASE_DIR, 'epoch=17-step=2178.ckpt'),
    os.path.join(BASE_DIR, '..', 'epoch=17-step=2178.ckpt'),
    'epoch=17-step=2178.ckpt'
]
weights_path = None
for p in weights_candidates:
    if os.path.exists(p):
        weights_path = p
        break

print(f"[*] Initializing Edge Inference Engine...")
if weights_path:
    print(f"[*] Found checkpoint: {weights_path}")
    engine = ModelInferenceEngine(weights_path)
else:
    print("[!] Warning: Checkpoint not found in standard paths. Initializing unweighted model.")
    engine = ModelInferenceEngine()

sim_manager = LiveSimulationManager(engine)
sample_data_dir = os.path.join(BASE_DIR, 'sample_data')
default_csv = os.path.join(sample_data_dir, 'CARE_Farm_C_SCADA_Full.csv')
replay_manager = DatasetReplayManager(engine, default_csv)

@app.route('/')
def index():
    """Renders the main dashboard."""
    return render_template('dashboard.html')

@app.route('/api/live/sample', methods=['GET'])
def get_live_sample():
    """Returns the latest single live SCADA sample and inference result."""
    sample = sim_manager.generate_next_sample()
    return jsonify(sample)

@app.route('/api/live/stream')
def stream_live_scada():
    """Server-Sent Events (SSE) stream for real-time telemetry."""
    def event_stream():
        while True:
            sample = sim_manager.generate_next_sample()
            yield f"data: {json.dumps(sample)}\n\n"
            time.sleep(1.0)
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/api/fault/inject', methods=['POST'])
def inject_fault():
    """Injects a chosen physical fault into the live turbine."""
    data = request.get_json() or {}
    fault_type = data.get('fault', 'bearing')
    res = sim_manager.inject_fault(fault_type)
    return jsonify(res)

@app.route('/api/fault/reset', methods=['POST'])
def reset_fault():
    """Resets turbine to normal baseline operation."""
    res = sim_manager.reset_fault()
    return jsonify(res)

@app.route('/api/turbine/shutdown', methods=['POST'])
def trigger_shutdown():
    """Triggers aerodynamic blade feathering and controlled emergency stop."""
    res = sim_manager.trigger_emergency_shutdown()
    return jsonify(res)

@app.route('/api/replay/sample', methods=['GET'])
def get_replay_sample():
    """Returns the next sequential SCADA sample from the dataset replay."""
    sample = replay_manager.get_next_sample()
    if sample is None:
        return jsonify({'error': 'No dataset available'}), 404
    return jsonify(sample)

@app.route('/api/replay/control', methods=['POST'])
def control_replay():
    """Handles replay playback controls: restart, jump, toggle_play."""
    data = request.get_json() or {}
    action = data.get('action', 'restart')
    if action in ['restart', 'reset']:
        res = replay_manager.restart()
    elif action in ['jump', 'next_fault', 'next']:
        res = replay_manager.jump_to_next_fault()
    elif action in ['toggle', 'toggle_play', 'play', 'pause']:
        res = replay_manager.toggle_play()
    else:
        res = {'status': 'unknown_action'}
    return jsonify(res)

@app.route('/api/replay/upload', methods=['POST'])
def upload_replay_file():
    """Uploads a SCADA file (.csv or .xlsx) and initiates immediate live streaming in Mode 2."""
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({'error': 'No file uploaded'}), 400
        uploaded_file = request.files['file']
        filename = uploaded_file.filename
        if not (filename.endswith('.csv') or filename.endswith('.xlsx') or filename.endswith('.xls')):
            return jsonify({'error': 'Unsupported format. Please upload .csv or .xlsx'}), 400
            
        save_path = os.path.join(sample_data_dir, filename)
        uploaded_file.save(save_path)
        replay_manager.load_dataset(save_path)
        return jsonify({'status': 'success', 'filename': filename, 'total_rows': len(replay_manager.df)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/samples', methods=['GET'])
def list_samples():
    """Lists available Farm C sample datasets."""
    files = []
    if os.path.exists(sample_data_dir):
        for f in os.listdir(sample_data_dir):
            if f.endswith('.csv') or f.endswith('.xlsx'):
                files.append(f)
    return jsonify({'samples': sorted(files)})

@app.route('/api/evaluate/file', methods=['POST'])
def evaluate_scada_file():
    """
    Evaluates an uploaded or selected SCADA dataset file (CSV or Excel).
    Runs zero-shot inference, computes CARE score, and returns timeline telemetry.
    """
    try:
        df = None
        filename = "Unknown"
        
        # Check if file uploaded via multipart
        if 'file' in request.files and request.files['file'].filename != '':
            uploaded_file = request.files['file']
            filename = uploaded_file.filename
            if filename.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(uploaded_file)
            else:
                return jsonify({'error': 'Unsupported format. Please upload .csv or .xlsx'}), 400
                
        # Check if pre-existing sample file requested
        elif request.is_json:
            sample_name = request.get_json().get('sample_file')
            if sample_name:
                filename = sample_name
                file_path = os.path.join(sample_data_dir, sample_name)
                if not os.path.exists(file_path):
                    return jsonify({'error': f'Sample file {sample_name} not found'}), 404
                if sample_name.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
        
        if df is None or len(df) == 0:
            return jsonify({'error': 'No data could be read from file'}), 400

        # Score dataset through model
        eval_result = engine.score_scada_dataframe(df)
        eval_result['filename'] = filename
        return jsonify(eval_result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/stats', methods=['GET'])
def get_system_stats():
    """Returns edge system performance metrics."""
    cpu_percent = psutil.cpu_percent(interval=None) if hasattr(psutil, 'cpu_percent') else 0.0
    mem = psutil.virtual_memory() if hasattr(psutil, 'virtual_memory') else None
    mem_used_mb = round(mem.used / 1024 / 1024, 1) if mem else 0.0
    mem_total_mb = round(mem.total / 1024 / 1024, 1) if mem else 0.0
    
    phys = engine.model.get_phys_params()
    
    return jsonify({
        'device': 'Raspberry Pi / Edge CPU',
        'cpu_usage_pct': cpu_percent,
        'ram_used_mb': mem_used_mb,
        'ram_total_mb': mem_total_mb,
        'model_parameters': 763572,
        'model_size_mb': 2.91,
        'threshold': engine.threshold,
        'crit_threshold': engine.crit_threshold,
        'physics_params': {
            'tau_oil_hrs': round(phys['tau_oil_hrs'], 1),
            'tau_brg_hrs': round(phys['tau_brg_hrs'], 1),
            'alpha_oil': round(phys['alpha_oil'], 6),
            'alpha_brg': round(phys['alpha_brg'], 6)
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*70}")
    print(f"  CARE WIND TURBINE PROGNOSTICS DASHBOARD")
    print(f"  Running on: http://localhost:{port}")
    print(f"  For Raspberry Pi local network access: http://<pi-ip-address>:{port}")
    print(f"{'='*70}\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
