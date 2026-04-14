import tensorflow as tf
import tensorflow_probability as tfp
import time
import os
import sys
from collections import namedtuple

class LogProbCounter:
    def __init__(self, log_prob_fn):
        self.log_prob_fn = log_prob_fn
        self.num_calls = tf.Variable(0, dtype=tf.int64, trainable=False)

    def __call__(self, *args):
        self.num_calls.assign_add(1)
        return self.log_prob_fn(*args)

MalaResults = namedtuple(
    "MalaResults",
    ["inner_results", "step_size"]
)

class MalaWithStepSize(tfp.mcmc.TransitionKernel):
    def __init__(self, target_log_prob_fn, step_size, volatility_fn):
        self._target_log_prob_fn = target_log_prob_fn
        self._volatility_fn = volatility_fn
        self._step_size = step_size

    @property
    def is_calibrated(self):
        return True

    def one_step(self, current_state, previous_kernel_results, seed=None):
        step_size = previous_kernel_results.step_size

        kernel = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=self._target_log_prob_fn,
            step_size=step_size,
            volatility_fn=self._volatility_fn,
        )

        new_state, inner_results = kernel.one_step(
            current_state,
            previous_kernel_results.inner_results,
            seed=seed
        )

        return new_state, MalaResults(
            inner_results=inner_results,
            step_size=step_size
        )

    def bootstrap_results(self, init_state):
        kernel = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=self._target_log_prob_fn,
            step_size=self._step_size,
            volatility_fn=self._volatility_fn,
        )

        inner_results = kernel.bootstrap_results(init_state)

        return MalaResults(
            inner_results=inner_results,
            step_size=tf.convert_to_tensor(self._step_size, tf.float32)
        )

def py_update(step, num_samples, num_burnin_steps, num_steps_between_results):
    global start_time
    if step == 0:
        start_time = time.time()
        return 0.0
    else:
        step = step // (num_steps_between_results + 1) + 1*int(num_steps_between_results > 0)

    step = int(step)
    start_time = float(start_time)
    burnin = float(num_burnin_steps)
    total = float(num_samples + num_burnin_steps)
    progress = step / total

    # --- PERCENT ---
    percent_value = int(progress * 100)
    percent = f"{percent_value}%"
    if len(percent) < 3:
        percent = "  " + percent
    elif len(percent) < 4:
        percent = " " + percent


    # --- COUNTER ---
    len_total = len(str(int(total)))
    len_step = len(str(step))
    diff_counter = len_total - len_step
    counter = " "*diff_counter + f"{step}/{int(total)}"

    # --- TIME ---
    elapsed = time.time() - start_time
    rate = max(step / max(elapsed, 1e-10), 1e-3)

    eta = (total - step) / rate

    # format time as mm:ss
    def format_time(seconds_total):
        minutes = int(seconds_total // 60)
        seconds = int(seconds_total % 60)

        minutes_str = str(minutes)
        seconds_str = str(seconds)

        # zero-pad manually
        if len(minutes_str) < 2:
            minutes_str = "0" + minutes_str
        if len(seconds_str) < 2:
            seconds_str = "0" + seconds_str

        return f"{minutes_str}:{seconds_str}"

    elapsed_str = format_time(elapsed)
    eta_str = format_time(eta)

    # --- RATE ---
    rate_value = round(rate * 100) / 100
    rate_str = f"{rate_value:.2f}"
    len_rate = len(rate_str)
    len_rate_max = 6
    diff_rate = len_rate_max - len_rate
    rate_str = " "*diff_rate + rate_str + " it/s"

    # --- BAR ---
    output_size = os.get_terminal_size().columns
    bar_width = max(output_size - len("".join([percent,counter,elapsed_str,eta_str,rate_str])) - 9, 10)
    filled = round(progress * bar_width)
    filled_burnin = round(min(burnin / total, progress) * bar_width)
    filled_sampling = filled - filled_burnin

    empty = bar_width - filled

    bar = "\033[93m█\033[0m" * filled_burnin + "█" * filled_sampling + " " * empty

    line = "".join([
            percent, "|",
            bar, "| ",
            counter,
            " [",
            elapsed_str, "<", eta_str, ", ",
            rate_str,
            "]"
    ])

    sys.stdout.write("\r" + line)
    sys.stdout.flush()
    if step == total:
        sys.stdout.write("\n")

    return 0.0


def trace_fn(_, pkr, num_samples, num_burnin_steps, inner_results, num_steps_between_results=0, progress_bar=True):
    if progress_bar:
        tf.py_function(
            func=lambda step: py_update(step, num_samples, num_burnin_steps, num_steps_between_results),
            inp=[pkr.step],
            Tout=tf.float32
        )
    else:
        if pkr.step > 0:
            tf.print("Step:", pkr.step, "of", num_samples + num_burnin_steps, " "*8, end="\r")
        if pkr.step == num_samples + num_burnin_steps - 1:
            tf.print()

    return (
        inner_results.is_accepted,
    )
