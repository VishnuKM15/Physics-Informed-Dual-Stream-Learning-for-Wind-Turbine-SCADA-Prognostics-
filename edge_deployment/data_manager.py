"""
Data Manager for CARE Wind Turbine Prognostics Dashboard.
Handles:
1. Real-time dynamic simulation with physics-constrained SCADA generation for Demo Mode.
2. Controllable fault injection & damage diagnosis with Staged Shutdown Logic.
3. Observation window management ("Keep running for a bit more" vs Immediate Critical Trip).
4. Continuous real-time SCADA telemetry logging console.
"""

import os
import time
import math
import random
import numpy as np
import pandas as pd

class LiveSimulationManager:
    """Simulates a live 3MW utility-scale wind turbine SCADA stream with staged diagnostic control."""
    def __init__(self, inference_engine):
        self.engine = inference_engine
        
        # Fault state
        self.active_fault = None
        self.fault_severity = 0.0
        self.fault_start_time = None
        self.step_count = 0
        
        # Staged decision engine state
        self.observation_active = False
        self.observation_timer = 0
        self.observation_stage = 1
        self.is_emergency_stop = False
        
        # Real-time console logs (holds last 50 entries)
        self.system_logs = []
        
        # State variables calibrated to physical nominal equilibrium
        self.wind_base = 12.8
        self.ambient_temp = 32.5
        self.oil_temp = 54.5
        self.bearing_temp = 60.5
        self.nacelle_temp = 41.0
        self.pitch_angle = 0.5
        
        # Initialize engine buffer with 36 steps of nominal history
        self._seed_healthy_buffer()
        self.add_log("System initialized. Edge inference engine loaded on Raspberry Pi CPU. Baseline optimal.")

    def add_log(self, msg):
        """Appends a timestamped log to the rolling console history."""
        timestamp = time.strftime('%I:%M:%S %p')
        log_entry = f"[{timestamp}] {msg}"
        self.system_logs.append(log_entry)
        if len(self.system_logs) > 50:
            self.system_logs.pop(0)

    def _seed_healthy_buffer(self):
        """Pre-populates the inference buffer with 36 nominal healthy samples."""
        self.engine.live_buffer = []
        for i in range(36):
            t_wave = math.sin(i * 0.15)
            w = 12.8 + 0.3 * t_wave
            rot = 12.9 + 0.1 * t_wave
            pwr = 1500.0 + 35.0 * t_wave
            pit = 0.5 + 0.02 * t_wave
            amb = 32.5 + 0.05 * math.cos(i * 0.1)
            oil = amb + 22.0 + 0.05 * t_wave
            brg = amb + 28.0 + 0.05 * t_wave
            feat = self.engine.raw_to_normalized(
                w, rot, pwr, pit, amb, oil, brg,
                brg_spread=2.3, p_res=0.0, elec=195.0, hyd=170.0, hyd_mask=1.0
            )
            self.engine.live_buffer.append(feat)

    def inject_fault(self, fault_type):
        """Triggers a physical fault with intelligent staged classification."""
        self.active_fault = fault_type
        self.fault_severity = 0.25
        self.fault_start_time = time.time()
        self.is_emergency_stop = False
        
        if fault_type == 'bearing':
            # Critical severity: immediate shutdown
            self.observation_active = False
            self.observation_stage = 2
            self.add_log(f"⚠️ FAULT INJECTED: Main Bearing Friction Surge. Severity: CRITICAL.")
            self.add_log(f"🚨 IMMEDIATE ACTION: Critical damage imminent! Initiating Emergency Trip & Blade Feathering.")
            
        elif fault_type in ['pitch', 'oil']:
            # Moderate to High severity: Staged hold ("keep running for a bit more")
            self.observation_active = True
            self.observation_timer = 20 # 20 seconds observation window
            self.observation_stage = 1
            if fault_type == 'pitch':
                self.add_log(f"⚠️ FAULT INJECTED: Pitch Mechanism Asymmetry. Severity: MODERATE.")
                self.add_log(f"⚙️ STAGE 1 DECISION: Non-critical damage. Keeping turbine running at de-rated 40% power.")
                self.add_log(f"⏱️ OBSERVATION WINDOW: 20s countdown active. Monitoring if aerodynamic fault persists.")
            else:
                self.add_log(f"⚠️ FAULT INJECTED: Gearbox Oil Cooler Pump Fault. Severity: HIGH.")
                self.add_log(f"⚙️ STAGE 1 DECISION: Auxiliary cooling booster engaged. De-rating turbine to 50% power.")
                self.add_log(f"⏱️ OBSERVATION WINDOW: 20s countdown active. Monitoring thermal dissipation curve.")
                
        else: # electrical
            self.observation_active = False
            self.observation_stage = 1
            self.add_log(f"⚠️ FAULT INJECTED: Power Converter Harmonic Imbalance. Severity: LOW (TRANSIENT).")
            self.add_log(f"ℹ️ DECISION: Do NOT shutdown. Minor electrical perturbation. Continuing normal generation.")
            
        return {'status': 'success', 'fault': fault_type}

    def reset_fault(self):
        """Restores healthy baseline immediately with clean reset."""
        self.active_fault = None
        self.fault_severity = 0.0
        self.fault_start_time = None
        self.observation_active = False
        self.observation_timer = 0
        self.observation_stage = 1
        self.is_emergency_stop = False
        self.step_count = 0
        
        # Hard reset all physics variables back to calibrated nominal
        self.wind_base = 12.8
        self.ambient_temp = 32.5
        self.oil_temp = 54.5
        self.bearing_temp = 60.5
        self.nacelle_temp = 41.0
        self.pitch_angle = 0.5
        
        self._seed_healthy_buffer()
        self.add_log("↺ SYSTEM REFRESH: Cleared all trip states. Restored turbine to healthy nominal baseline (13.0 RPM, 1500 kW).")
        return {'status': 'success', 'fault': 'none'}

    def trigger_emergency_shutdown(self):
        """Simulates manual operator emergency shutdown."""
        self.is_emergency_stop = True
        self.observation_active = False
        self.observation_stage = 2
        self.add_log("🛑 MANUAL EMERGENCY SHUTDOWN: Operator command executed. Blades feathering to 90°.")
        return {'status': 'success', 'action': 'shutdown'}

    def generate_next_sample(self):
        """
        Generates the next second's SCADA packet.
        Applies wind turbine aerodynamics, thermodynamics, and staged decision state.
        """
        self.step_count += 1
        t = self.step_count
        
        # Environmental inflow
        wind_gust = 0.4 * math.sin(t * 0.15) + 0.2 * math.sin(t * 0.4) + random.uniform(-0.1, 0.1)
        wind = max(3.0, self.wind_base + wind_gust)
        ambient = self.ambient_temp + 0.1 * math.sin(t * 0.02) + random.uniform(-0.02, 0.02)
        
        # Handle Staged Response Observation Countdown
        if self.observation_active and self.observation_timer > 0:
            self.observation_timer -= 1
            if self.observation_timer in [15, 10, 5]:
                self.add_log(f"⏱️ OBSERVATION T-{self.observation_timer}s: Turbine running de-rated. Checking if fault clears...")
            elif self.observation_timer == 0:
                # Observation period ended without recovery -> Escalate to Stage 2 Emergency Trip
                self.observation_active = False
                self.observation_stage = 2
                self.is_emergency_stop = True
                self.add_log(f"🛑 STAGE 2 ESCALATION: 20s observation window expired! Fault persistent. Initiating Controlled Shutdown & Feathering.")

        # Emergency Shutdown / Braking Dynamics
        if self.is_emergency_stop:
            pitch = min(90.0, 0.5 + (t % 100) * 15.0)
            rotor = max(0.0, 13.0 - (t % 100) * 2.5)
            power = max(0.0, 1500.0 - (t % 100) * 400.0)
        elif self.observation_active:
            # Stage 1: De-rated power generation while observing
            pitch = max(0.0, 3.5 - 0.1 * wind_gust + random.uniform(-0.05, 0.05))
            rotor = max(6.0, 10.5 + 0.1 * wind_gust)
            power = max(200.0, 650.0 + 30.0 * wind_gust + random.uniform(-10.0, 10.0))
        else:
            # Full Nominal Power Generation
            pitch = max(0.0, 0.5 - 0.1 * wind_gust + random.uniform(-0.05, 0.05))
            rotor = max(0.0, 13.0 + 0.15 * wind_gust + random.uniform(-0.05, 0.05))
            power = max(0.0, min(3000.0, 1500.0 + 55.0 * wind_gust + random.uniform(-10.0, 10.0)))
            
        # Nominal thermal rise based on model ODE
        p_pu = power / 3000.0
        p_params = self.engine.model.get_phys_params()
        
        oil_rise_nom = (1.0 - p_params['beta_oil']) * (self.oil_temp - ambient) / 100.0 + p_params['alpha_oil'] * p_pu
        brg_rise_nom = (1.0 - p_params['beta_brg']) * (self.bearing_temp - ambient) / 100.0 + p_params['alpha_brg'] * p_pu
        
        self.oil_temp = ambient + oil_rise_nom * 100.0 + random.uniform(-0.03, 0.03)
        self.bearing_temp = ambient + brg_rise_nom * 100.0 + random.uniform(-0.03, 0.03)
        self.nacelle_temp = ambient + 7.5 + random.uniform(-0.05, 0.05)

        # APPLY PHYSICAL FAULT EFFECTS
        fault_name = "None"
        if self.active_fault:
            self.fault_severity = min(1.0, self.fault_severity + 0.05)
            sev = self.fault_severity
            
            if self.active_fault == 'bearing':
                fault_name = "Main Bearing Spalling & Friction Surge"
                self.bearing_temp += 42.0 * sev
                if not self.is_emergency_stop:
                    self.is_emergency_stop = True # Immediate trip on critical bearing fault
                
            elif self.active_fault == 'oil':
                fault_name = "Gearbox Oil Cooler Pump Failure"
                self.oil_temp += 45.0 * sev
                self.nacelle_temp += 16.0 * sev
                
            elif self.active_fault == 'pitch':
                fault_name = "Pitch Actuator Drift & Aerodynamic Stall"
                if not self.is_emergency_stop:
                    pitch += 22.0 * sev
                    power = max(100.0, power - 850.0 * sev)
                
            elif self.active_fault == 'electrical':
                fault_name = "Converter IGBT Phase Oscillation"
                power = max(50.0, power + 400.0 * math.sin(t * 1.5) * sev)

        # Calculate secondary physical SCADA channels matching model training
        gen_speed = round(rotor * 95.1 + random.uniform(-1.0, 1.0), 1)
        reactive = round(max(0.0, 75.0 + 5.0 * (power / 1500.0) + random.uniform(-2.0, 2.0)), 1)
        brg_spread = round(2.3 + (18.5 * sev if self.active_fault == 'bearing' else 0.0) + random.uniform(-0.1, 0.1), 2)
        power_res = round((-40.0 * sev if self.active_fault in ['bearing', 'pitch'] else 0.0) + random.uniform(-4.0, 4.0), 1)
        current = round(max(0.0, (power * 1000.0) / (math.sqrt(3) * 690.0 * 0.95) + random.uniform(-2.0, 2.0)), 1)
        hydraulic = round(max(0.0, 170.0 - (16.0 * sev if self.active_fault == 'pitch' else 0.0) + random.uniform(-1.5, 1.5)), 1)

        raw_sample = {
            'wind': round(wind, 2),
            'rotor': round(rotor, 2),
            'gen_speed': gen_speed,
            'power': round(power, 1),
            'reactive': reactive,
            'pitch': round(pitch, 2),
            'amb': round(ambient, 2),
            'oil': round(self.oil_temp, 2),
            'brg': round(self.bearing_temp, 2),
            'brg_spread': brg_spread,
            'p_res': power_res,
            'elec': current,
            'hyd': hydraulic,
            'hyd_mask': 1.0,
            'nacelle': round(self.nacelle_temp, 2),
            'timestamp': time.strftime('%H:%M:%S'),
            'fault_injected': fault_name,
            'active_fault_type': self.active_fault,
            'is_shutdown': self.is_emergency_stop,
            'observation_active': self.observation_active,
            'observation_timer': self.observation_timer,
            'observation_stage': self.observation_stage
        }
        
        # Run Edge Model Inference & Diagnostic Classification
        infer_result = self.engine.process_live_sample(raw_sample)
        
        # Formulate formatted SCADA logging message
        scada_log = (f"SCADA: Wind={raw_sample['wind']:.1f}m/s | Rotor={raw_sample['rotor']:.1f}RPM | Gen={raw_sample['gen_speed']:.0f}RPM | "
                     f"Power={raw_sample['power']:.0f}kW | T_oil={raw_sample['oil']:.1f}°C | "
                     f"T_brg={raw_sample['brg']:.1f}°C (Spread:{raw_sample['brg_spread']:.1f}°C) | "
                     f"Current={raw_sample['elec']:.0f}A | Hyd={raw_sample['hyd']:.0f}bar | Score={infer_result['anomaly_score']:.2f}")
        
        # Append SCADA telemetry log periodically (every step)
        self.add_log(scada_log)
        
        output = {
            **raw_sample,
            **infer_result,
            'system_logs': list(self.system_logs),
            'observation_timer': self.observation_timer,
            'observation_active': self.observation_active,
            'observation_stage': self.observation_stage
        }
        return output


class DatasetReplayManager:
    """
    Manages sequential streaming / replay of a SCADA dataset file.
    Streams row-by-row through the PyTorch PINN-AE edge model with real-time logging,
    fault attribution, and staged decision dynamics.
    """
    def __init__(self, inference_engine, default_csv_path=None):
        self.engine = inference_engine
        self.file_path = default_csv_path
        self.df = None
        self.current_idx = 0
        self.is_playing = True
        self.playback_speed = 1.0
        self.is_shutdown = False
        self.trip_hold_counter = 0
        self.replay_logs = []
        
        if default_csv_path and os.path.exists(default_csv_path):
            self.load_dataset(default_csv_path)

    def load_dataset(self, file_path_or_df):
        if isinstance(file_path_or_df, str):
            self.file_path = file_path_or_df
            if file_path_or_df.endswith('.csv'):
                self.df = pd.read_csv(file_path_or_df)
            else:
                self.df = pd.read_excel(file_path_or_df)
        elif isinstance(file_path_or_df, pd.DataFrame):
            self.df = file_path_or_df
        
        self.current_idx = 0
        self.is_shutdown = False
        self.trip_hold_counter = 0
        self.replay_logs = []
        self._seed_replay_buffer()
        self.add_log("Farm C SCADA dataset loaded. Streaming initiated.")

    def add_log(self, msg):
        timestamp = time.strftime('%I:%M:%S %p')
        entry = f"[{timestamp}] {msg}"
        self.replay_logs.append(entry)
        if len(self.replay_logs) > 50:
            self.replay_logs.pop(0)

    def _seed_replay_buffer(self):
        """Initializes rolling buffer using first 36 rows of the dataset."""
        self.engine.live_buffer = []
        if self.df is not None and len(self.df) >= 36:
            for i in range(36):
                row = self.df.iloc[i]
                w = float(row.get('wind_speed_ms', 12.8))
                rot = float(row.get('rotor_speed_rpm', 12.9))
                pwr = float(row.get('active_power_kw', 1500.0))
                pit = float(row.get('pitch_angle_deg', 0.5))
                amb = float(row.get('ambient_temp_c', 32.5))
                oil = float(row.get('gearbox_oil_temp_c', 54.5))
                brg = float(row.get('main_bearing_temp_c', 60.5))
                brg_spread = float(row.get('bearing_spread_c', 2.3))
                p_res = float(row.get('power_residual_kw', 0.0))
                current = float(row.get('phase_current_rms_a', 195.0))
                hydraulic = float(row.get('hydraulic_pressure_bar', 170.0))
                hyd_mask = float(row.get('hydraulic_mask', 1.0))
                feat = self.engine.raw_to_normalized(
                    w, rot, pwr, pit, amb, oil, brg,
                    brg_spread=brg_spread, p_res=p_res, elec=current, hyd=hydraulic, hyd_mask=hyd_mask
                )
                self.engine.live_buffer.append(feat)
            self.current_idx = 36

    def restart(self):
        self.current_idx = 0
        self.is_shutdown = False
        self.trip_hold_counter = 0
        self._seed_replay_buffer()
        self.add_log("↺ REPLAY RESTARTED: Re-seeded buffer to dataset beginning.")
        return {'status': 'success', 'current_idx': self.current_idx}

    def toggle_play(self):
        self.is_playing = not self.is_playing
        state_txt = "Resumed" if self.is_playing else "Paused"
        self.add_log(f"⏸️ REPLAY CONTROL: Stream {state_txt} by operator.")
        return {'status': 'success', 'is_playing': self.is_playing}

    def jump_to_next_fault(self):
        """Fast-forwards to the next fault episode in the dataset."""
        if self.df is None:
            return {'status': 'error', 'msg': 'No dataset loaded'}
        status_col = 'status_type_id' if 'status_type_id' in self.df.columns else None
        if status_col:
            fault_indices = self.df.index[(self.df.index > self.current_idx + 10) & (~self.df[status_col].isin([0, 2]))].tolist()
            if fault_indices:
                target = max(36, fault_indices[0] - 15)
                self.current_idx = target
                self.is_shutdown = False
                self.trip_hold_counter = 0
                asset = self.df.iloc[target].get('asset_id', 'Turbine')
                self.add_log(f"⏭️ FAST-FORWARD: Jumped to fault lead-up sequence in {asset}.")
                return {'status': 'success', 'jumped_to': target, 'asset': asset}
        return self.restart()

    def get_next_sample(self):
        """Extracts the next row, runs PyTorch inference, handles physical reactions and logging."""
        if self.df is None or len(self.df) == 0:
            return None
            
        if self.current_idx >= len(self.df):
            self.current_idx = 0
            self.add_log("↺ End of dataset reached. Looping back to beginning.")
            
        row = self.df.iloc[self.current_idx]
        if self.is_playing:
            self.current_idx += 1
            
        asset_id = str(row.get('asset_id', 'Turbine_53'))
        ts_str = str(row.get('time_stamp', time.strftime('%Y-%m-%d %H:%M:%S')))
        status_id = int(row.get('status_type_id', 0))
        
        # Raw SCADA values from complete consolidated dataset
        wind = float(row.get('wind_speed_ms', 12.8))
        rotor = float(row.get('rotor_speed_rpm', 12.9))
        gen = float(row.get('generator_speed_rpm', rotor * 95.1))
        power = float(row.get('active_power_kw', 1500.0))
        reactive = float(row.get('reactive_power_kvar', 75.0))
        pitch = float(row.get('pitch_angle_deg', 0.5))
        amb = float(row.get('ambient_temp_c', 32.5))
        oil = float(row.get('gearbox_oil_temp_c', 54.5))
        brg = float(row.get('main_bearing_temp_c', 60.5))
        brg_spread = float(row.get('bearing_spread_c', 2.3))
        power_res = float(row.get('power_residual_kw', 0.0))
        current = float(row.get('phase_current_rms_a', 195.0))
        hydraulic = float(row.get('hydraulic_pressure_bar', 170.0))
        hyd_mask = float(row.get('hydraulic_mask', 1.0))
        nacelle = float(row.get('nacelle_temp_c', amb + 8.5))
        
        # Run PyTorch edge model inference
        raw_sample = {
            'wind': round(wind, 2),
            'rotor': round(rotor, 2),
            'gen_speed': round(gen, 1),
            'power': round(power, 1),
            'reactive': round(reactive, 1),
            'pitch': round(pitch, 2),
            'amb': round(amb, 2),
            'oil': round(oil, 2),
            'brg': round(brg, 2),
            'brg_spread': round(brg_spread, 2),
            'p_res': round(power_res, 1),
            'elec': round(current, 1),
            'hyd': round(hydraulic, 1),
            'hyd_mask': round(hyd_mask, 1),
            'nacelle': round(nacelle, 2),
            'timestamp': ts_str,
            'asset_id': asset_id,
            'status_type_id': status_id
        }
        infer = self.engine.process_live_sample(raw_sample)
        score = infer['anomaly_score']
        
        now_time = time.strftime('%I:%M:%S %p')
        # Log SCADA packet with real-time timestamp
        scada_row_log = (f"[{now_time}] {asset_id} | SCADA: Wind={wind:.1f}m/s, Rotor={rotor:.1f}rpm, Gen={gen:.0f}rpm, "
                         f"Power={power:.0f}kW, Pitch={pitch:.1f}°, Oil={oil:.1f}°C, Brg={brg:.1f}°C (Spread:{brg_spread:.1f}°C), "
                         f"Current={current:.0f}A, Hyd={hydraulic:.0f}bar | Score={score:.2f}")
        self.add_log(scada_row_log)
        
        # Staged decision dynamics for dataset replay
        if score >= self.engine.crit_threshold or status_id == 5:
            self.is_shutdown = True
            self.trip_hold_counter += 1
            rotor_disp = max(0.0, 12.8 * math.exp(-self.trip_hold_counter / 2.0))
            pitch_disp = min(90.0, 0.5 + (90.0 - 0.5) * (1.0 - math.exp(-self.trip_hold_counter / 2.0)))
            power_disp = max(0.0, power * math.exp(-self.trip_hold_counter / 2.0))
            
            if self.trip_hold_counter == 1:
                self.add_log(f"🚨 CRITICAL TRIP [{asset_id}]: Thermal safety boundary breached! Score={score:.2f} > 4.50.")
                self.add_log(f"🛑 ACTION TAKEN: Controlled emergency stop deployed. Blades feathered to 90.0°, mechanical brake engaged.")
            elif self.trip_hold_counter == 5:
                self.add_log(f"ℹ️ [{asset_id}] Turbine halted safely at 0.0 RPM. Drivetrain protected. Click 'Next Fault Event' to advance.")
                
        elif score >= self.engine.threshold or status_id in [3, 9, 12]:
            self.is_shutdown = False
            self.trip_hold_counter = 0
            rotor_disp = max(6.0, rotor * 0.8)
            pitch_disp = pitch + 3.0
            power_disp = power * 0.6
            if (self.current_idx % 10) == 0:
                self.add_log(f"⚠️ ANOMALY DETECTED [{asset_id}]: Score={score:.2f} > 2.81 | Subsystem: {infer['damage_class']}")
        else:
            self.is_shutdown = False
            self.trip_hold_counter = 0
            rotor_disp = rotor
            pitch_disp = pitch
            power_disp = power

        return {
            'wind': round(wind, 2),
            'rotor': round(rotor_disp, 2),
            'gen_speed': round(gen, 1),
            'power': round(power_disp, 1),
            'reactive': round(reactive, 1),
            'pitch': round(pitch_disp, 2),
            'amb': round(amb, 2),
            'oil': round(oil, 2),
            'brg': round(brg, 2),
            'brg_spread': round(brg_spread, 2),
            'power_res': round(power_res, 1),
            'current': round(current, 1),
            'hydraulic': round(hydraulic, 1),
            'nacelle': round(nacelle, 2),
            'timestamp': ts_str,
            'asset_id': asset_id,
            'status_type_id': status_id,
            'is_shutdown': self.is_shutdown,
            'current_row': self.current_idx,
            'total_rows': len(self.df),
            'is_playing': self.is_playing,
            'system_logs': list(self.replay_logs),
            **infer
        }
