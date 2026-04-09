import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# HODGKIN-HUXLEY NEURON — Layer 3
# =============================================================================
# No phases. No timers. No magic numbers for spike shape.
# Four coupled differential equations. Spike emerges from conductances.
#
# State variables:
#   vm — membrane potential (mV)
#   m  — Na+ activation gate      (0→1, fast, opens with depolarization)
#   h  — Na+ inactivation gate    (1→0, slow, closes with depolarization)
#   n  — K+  activation gate      (0→1, slow, opens with depolarization)
#
# Conductances:
#   gNa = G_Na_max * m³ * h       (m and h are opponents — spike lives in the window)
#   gK  = G_K_max  * n⁴
#
# Membrane equation (KCL):
#   C * dvm/dt = I_input - gNa*(vm-E_Na) - gK*(vm-E_K) - gL*(vm-E_L)
#
# Gate equation (population balance):
#   dx/dt = α(vm)*(1-x) - β(vm)*x
#   equivalently: dx/dt = (x∞ - x) / τ
#   where x∞ = α/(α+β),  τ = 1/(α+β)

class HodgkinHuxley:
    def __init__(
        self,
        # membrane
        C          = 1.0,    # capacitance (μF/cm²)
        # max conductances (mS/cm²)
        G_Na_max   = 120.0,
        G_K_max    = 36.0,
        G_L        = 0.3,
        # reversal potentials (mV)
        E_Na       = 55.0,
        E_K        = -77.0,
        E_L        = -65.0,
        # simulation
        dt         = 0.01,   # ms — HH needs fine resolution
        total_time = 50.0,   # ms
    ):
        self.C          = C
        self.G_Na_max   = G_Na_max
        self.G_K_max    = G_K_max
        self.G_L        = G_L
        self.E_Na       = E_Na
        self.E_K        = E_K
        self.E_L        = E_L
        self.dt         = dt
        self.total_time = total_time

        # initialize state at resting potential
        self.vm = -65.0
        self.m, self.h, self.n = self._steady_state(self.vm)

    # -------------------------------------------------------------------------
    # alpha and beta functions — HH curve fits to experimental data
    # these encode how fast gates open/close at each voltage
    # -------------------------------------------------------------------------

    def _alpha_m(self, vm):
        # singularity at vm = -40, use L'Hopital limit
        if abs(vm + 40) < 1e-7:
            return 1.0
        return 0.1 * (vm + 40) / (1 - np.exp(-(vm + 40) / 10))

    def _beta_m(self, vm):
        return 4.0 * np.exp(-(vm + 65) / 18)

    def _alpha_h(self, vm):
        return 0.07 * np.exp(-(vm + 65) / 20)

    def _beta_h(self, vm):
        return 1.0 / (1 + np.exp(-(vm + 35) / 10))

    def _alpha_n(self, vm):
        # singularity at vm = -55, use L'Hopital limit
        if abs(vm + 55) < 1e-7:
            return 0.1
        return 0.01 * (vm + 55) / (1 - np.exp(-(vm + 55) / 10))

    def _beta_n(self, vm):
        return 0.125 * np.exp(-(vm + 65) / 80)

    # -------------------------------------------------------------------------
    # steady state values — where gates sit at a given voltage
    # used to initialize m, h, n so we don't start mid-spike
    # -------------------------------------------------------------------------

    def _steady_state(self, vm):
        am, bm = self._alpha_m(vm), self._beta_m(vm)
        ah, bh = self._alpha_h(vm), self._beta_h(vm)
        an, bn = self._alpha_n(vm), self._beta_n(vm)
        m_inf = am / (am + bm)
        h_inf = ah / (ah + bh)
        n_inf = an / (an + bn)
        return m_inf, h_inf, n_inf

    # -------------------------------------------------------------------------
    # single time step — Euler integration of all four equations
    # -------------------------------------------------------------------------

    def step(self, I_input):
        vm, m, h, n = self.vm, self.m, self.h, self.n

        # gate equations: dx/dt = α*(1-x) - β*x
        am, bm = self._alpha_m(vm), self._beta_m(vm)
        ah, bh = self._alpha_h(vm), self._beta_h(vm)
        an, bn = self._alpha_n(vm), self._beta_n(vm)

        dm = (am * (1 - m) - bm * m) * self.dt
        dh = (ah * (1 - h) - bh * h) * self.dt
        dn = (an * (1 - n) - bn * n) * self.dt

        # conductances — emerge from gate states
        gNa = self.G_Na_max * m**3 * h
        gK  = self.G_K_max  * n**4

        # currents — each ion pulling vm toward its reversal potential
        I_Na = gNa * (vm - self.E_Na)
        I_K  = gK  * (vm - self.E_K)
        I_L  = self.G_L * (vm - self.E_L)

        # membrane equation
        dvm = (I_input - I_Na - I_K - I_L) / self.C * self.dt

        # update state
        self.vm += dvm
        self.m  += dm
        self.h  += dh
        self.n  += dn

        return {
            "vm"  : self.vm,
            "m"   : self.m,
            "h"   : self.h,
            "n"   : self.n,
            "gNa" : gNa,
            "gK"  : gK,
            "I_Na": I_Na,
            "I_K" : I_K,
            "I_L" : I_L,
        }

    # -------------------------------------------------------------------------
    # reset — back to resting state, ready to simulate again
    # -------------------------------------------------------------------------

    def reset(self):
        self.vm = -65.0
        self.m, self.h, self.n = self._steady_state(self.vm)


# =============================================================================
# SIMULATION — owns time, collects history, knows nothing about plotting
# =============================================================================

def simulate(neuron, input_current_fn):
    """
    neuron          — HodgkinHuxley instance
    input_current_fn — function: f(t) → current (μA/cm²)
                       gives you full control over input shape
    """
    neuron.reset()

    steps        = int(neuron.total_time / neuron.dt)
    current_time = 0.0

    history = {
        "t"   : [],
        "vm"  : [],
        "m"   : [],
        "h"   : [],
        "n"   : [],
        "gNa" : [],
        "gK"  : [],
        "I_Na": [],
        "I_K" : [],
        "I_L" : [],
        "I_in": [],
    }

    for _ in range(steps):
        I_input = input_current_fn(current_time)
        state   = neuron.step(I_input)

        history["t"].append(current_time)
        history["I_in"].append(I_input)
        for key, val in state.items():
            history[key].append(val)

        current_time += neuron.dt

    # convert to numpy for easy slicing
    return {k: np.array(v) for k, v in history.items()}


# =============================================================================
# VISUALIZATION — modular, call any combination independently
# =============================================================================

def plot_vm(history, ax=None, title="membrane potential"):
    ax = ax or plt.gca()
    ax.plot(history["t"], history["vm"], color="steelblue", linewidth=1.5)
    ax.axhline(-55, color="red",  linestyle="--", linewidth=0.8, alpha=0.5, label="threshold (−55)")
    ax.axhline(-65, color="gray", linestyle="--", linewidth=0.8, alpha=0.4, label="resting (−65)")
    ax.set_ylabel("mV")
    ax.set_title(title)
    ax.legend(fontsize=8)


def plot_gates(history, ax=None):
    ax = ax or plt.gca()
    ax.plot(history["t"], history["m"], color="coral",      linewidth=1.2, label="m  (Na⁺ activation, fast)")
    ax.plot(history["t"], history["h"], color="steelblue",  linewidth=1.2, label="h  (Na⁺ inactivation, slow)")
    ax.plot(history["t"], history["n"], color="mediumseagreen", linewidth=1.2, label="n  (K⁺ activation, slow)")
    ax.set_ylabel("gate value (0–1)")
    ax.set_title("gating variables")
    ax.legend(fontsize=8)


def plot_conductances(history, ax=None):
    ax = ax or plt.gca()
    ax.plot(history["t"], history["gNa"], color="coral",         linewidth=1.2, label="gNa = G_max·m³·h")
    ax.plot(history["t"], history["gK"],  color="mediumseagreen", linewidth=1.2, label="gK  = G_max·n⁴")
    ax.set_ylabel("conductance (mS/cm²)")
    ax.set_title("conductances")
    ax.legend(fontsize=8)


def plot_currents(history, ax=None):
    ax = ax or plt.gca()
    ax.plot(history["t"], history["I_Na"], color="coral",         linewidth=1.2, label="I_Na")
    ax.plot(history["t"], history["I_K"],  color="mediumseagreen", linewidth=1.2, label="I_K")
    ax.plot(history["t"], history["I_L"],  color="gray",          linewidth=1.2, label="I_L")
    ax.plot(history["t"], history["I_in"], color="black",         linewidth=1.0,
            linestyle="--", alpha=0.6, label="I_input")
    ax.set_ylabel("current (μA/cm²)")
    ax.set_title("currents")
    ax.legend(fontsize=8)


def plot_all(history):
    """full four-panel view — the standard HH diagnostic plot"""
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    plot_vm(history,           ax=axes[0])
    plot_gates(history,        ax=axes[1])
    plot_conductances(history, ax=axes[2])
    plot_currents(history,     ax=axes[3])
    axes[-1].set_xlabel("time (ms)")
    plt.tight_layout()
    plt.show()


# =============================================================================
# INPUT FUNCTIONS — swap these to run different experiments
# =============================================================================

def pulse(start=10, end=11, strength=10):
    """single brief pulse — produces one spike"""
    return lambda t: strength if start <= t <= end else 0.0

def step_current(start=10, strength=10):
    """constant current from start onward — produces repeated firing"""
    return lambda t: strength if t >= start else 0.0

def double_pulse(t1=10, t2=20, strength=10):
    """two pulses — good for testing refractory period"""
    return lambda t: strength if (t1 <= t <= t1+1 or t2 <= t <= t2+1) else 0.0
