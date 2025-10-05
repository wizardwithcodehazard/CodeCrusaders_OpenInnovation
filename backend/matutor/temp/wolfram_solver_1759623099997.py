import json
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

session = None  # Initialize session to None
results = {}

try:
    session = WolframLanguageSession()

    # Wolfram Language script to perform all calculations
    # Variables are defined and used sequentially within this block.
    # The final expression is a list of all required numerical results.
    calculations_script = wlexpr("""
        (* Transformer Parameters *)
        Np = 1000; (* Primary turns *)
        Ns = 200;  (* Secondary turns *)
        
        (* No-load conditions *)
        noLoadCurrentMagnitude = 3; (* Amperes *)
        noLoadPowerFactor = 0.2;    (* lag *)
        
        (* Secondary load conditions *)
        secondaryCurrentMagnitude = 280; (* Amperes *)
        secondaryPowerFactor = 0.8;      (* lag *)
        
        (* --- Calculations --- *)
        
        (* 1. Magnetizing component and loss component of no-load current *)
        
        (* No-load angle (phi0) from power factor: ArcCos[PF] *)
        noLoadAngle = ArcCos[noLoadPowerFactor];
        
        (* Loss component (Iw) - in phase with voltage *)
        lossComponent = noLoadCurrentMagnitude * noLoadPowerFactor;
        
        (* Magnetizing component (Im) - quadrature to voltage *)
        (* For lagging power factor, Sin[noLoadAngle] is positive *)
        magnetizingComponent = noLoadCurrentMagnitude * Sin[noLoadAngle];
        
        (* 2. Primary current *)
        
        (* Secondary load angle (phi2) from power factor: ArcCos[PF] *)
        secondaryAngle = ArcCos[secondaryPowerFactor];
        
        (* Referred secondary current magnitude to primary side: I2' = I2 * (Ns/Np) *)
        secondaryReferredMagnitude = secondaryCurrentMagnitude * (Ns/Np);
        
        (* No-load current phasor (I0) *)
        (* Assuming voltage is reference (0 degrees), current lags by angle phi0 *)
        (* I0_phasor = I0_mag * (Cos[phi0] - I * Sin[phi0]) for lagging *)
        noLoadPhasor = noLoadCurrentMagnitude * (Cos[noLoadAngle] - I * Sin[noLoadAngle]);
        
        (* Referred secondary current phasor (I2') *)
        (* I2'_phasor = I2'_mag * (Cos[phi2] - I * Sin[phi2]) for lagging *)
        secondaryReferredPhasor = secondaryReferredMagnitude * (Cos[secondaryAngle] - I * Sin[secondaryAngle]);
        
        (* Primary current (I1) is the phasor sum of no-load current and referred secondary current *)
        primaryCurrentPhasor = noLoadPhasor + secondaryReferredPhasor;
        
        (* Magnitude of the primary current *)
        primaryCurrentMagnitude = Abs[primaryCurrentPhasor];
        
        (* 3. Input power factor *)
        
        (* Input power factor is Cos[Angle of the primary current phasor] *)
        (* Argument[Z] gives the angle of complex number Z *)
        inputPowerFactor = Cos[Argument[primaryCurrentPhasor]];
        
        (* Return all calculated values as a list. Order matters for Python conversion. *)
        {magnetizingComponent, lossComponent, primaryCurrentMagnitude, inputPowerFactor}
    """)

    # Evaluate the entire script within Wolfram Engine.
    # wl.N() is CRITICAL to force numerical evaluation of all expressions.
    raw_results = session.evaluate(wl.N(calculations_script))
    
    # Convert Wolfram Language list output to Python list of floats
    if isinstance(raw_results, list) and len(raw_results) == 4:
        magnetizing_component = float(raw_results[0])
        loss_component = float(raw_results[1])
        primary_current = float(raw_results[2])
        input_power_factor = float(raw_results[3])
    else:
        raise ValueError(f"Unexpected result format from Wolfram Engine: {raw_results}")

    results = {
        "magnetizing_component_no_load_current_A": magnetizing_component,
        "loss_component_no_load_current_A": loss_component,
        "primary_current_A": primary_current,
        "input_power_factor": input_power_factor
    }

    print(json.dumps(results))

except Exception as e:
    results = {"error": str(e)}
    print(json.dumps(results))

finally:
    # Ensure the Wolfram Language session is terminated
    if session is not None:
        session.terminate()