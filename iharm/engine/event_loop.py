import time
import math
from collections import deque


class EventLoop(object):
    def __init__(self, console_logger=None, tb_writer=None, period_length=1000, start_tick=0,
                 speed_counter='ma_10', speed_tb_period=-1):
        self.period_length = period_length
        self.total_ticks = start_tick
        self.events = []
        self.metrics = {}
        self.last_time = None
        self.tb_writer = tb_writer
        self.console_logger = console_logger
        self._speed_metric = '_speed_'
        self._max_delta_time = 60
        self._prev_total_ticks = 0

        self.register_metric(
            self._speed_metric,
            console_name='Speed', console_period=-1, console_format='{:.2f} samples/s',
            tb_name='Misc/Speed', tb_period=speed_tb_period,
            counter=speed_counter
        )

    def register_event(self, event, period, func_inputs=(),
                       last_step=0, associated_metric=None, onetime=False):
        event = {
            'event': event,
            'last_step': last_step,
            'period': period,
            'func_inputs': func_inputs,
            'onetime': onetime
        }
        if associated_metric is not None:
            event['metric'] = associated_metric

        self.events.append(event)

    def register_metric(self, metric_name,
                        console_name=None, console_period=-1, console_format='{:.3f}',
                        tb_name=None, tb_period=-1, tb_global_step='n_ticks',
                        counter=None):

        self.metrics[metric_name] = {
            'console_name': console_name if console_name is not None else metric_name,
            'console_event': {'last_step': 0, 'period': console_period},
            'console_format': console_format,
            'tb_name': tb_name if tb_name is not None else metric_name,
            'tb_event': {'last_step': 0, 'period': tb_period},
            'counters': {counter: parse_counter(counter)},
            'default_counter': counter,
            'tb_global_step': tb_global_step
        }

    def register_metric_event(self, event, metric_name, period, func_inputs=(),
                              console_period=0, tb_period=0,
                              **metric_kwargs):
        self.register_event(event, period, func_inputs=func_inputs, associated_metric=metric_name)
        self.register_metric(metric_name, console_period=console_period, tb_period=tb_period,
                             **metric_kwargs)

    def add_metric_value(self, metric_name, value):
        metric = self.metrics[metric_name]
        for counter in metric['counters'].values():
            counter.add(value)
        metric['console_event']['relaxed'] = False
        metric['tb_event']['relaxed'] = False

    def add_custom_metric_counter(self, metric_name, counter):
        metric = self.metrics[metric_name]
        if counter not in metric['counters']:
            metric['counters'][counter] = parse_counter(counter)

    def get_metric_value(self, metric_name, counter_name=None):
        metric = self.metrics[metric_name]
        if counter_name is None:
            return metric['counters'][metric['default_counter']].value
        else:
            return metric['counters'][counter_name].value

    def step(self, step_size):
        self._prev_total_ticks = self.total_ticks
        self.total_ticks += step_size

        self._update_time(step_size)
        self._check_events()
        self._check_metrics()

    def get_states(self):
        return {
            'metrics': self.metrics,
            'total_ticks': self.total_ticks,
            '_prev_total_ticks': self._prev_total_ticks,
            'base_period': self.period_length
        }

    def set_states(self, states):
        for k, v in states.items():
            if k == 'metrics':
                self.metrics.update(v)
            else:
                setattr(self, k, v)

    @property
    def n_periods(self):
        return self.total_ticks // self.period_length

    @property
    def f_periods(self):
        return self.total_ticks / self.period_length

    @property
    def n_ticks(self):
        return self.total_ticks

    def _check_events(self):
        triggered_events = []
        for event in self.events:
            if self._check_event(event, ignore_relaxed=True):
                triggered_events.append(event)
                if event['onetime']:
                    event['remove'] = True

        self.events = [event for event in self.events if not event.get('remove', False)]

        for event in triggered_events:
            event_metric = event.get('metric', None)
            inputs = (getattr(self, input_name) for input_name in event['func_inputs'])
            if event_metric is None:
                event['event'](*inputs)
            else:
                metric_value = event['event'](*inputs)
                self.add_metric_value(event_metric, metric_value)

    def _check_metrics(self):
        print_list = []
        for metric_name, metric in self.metrics.items():
            if self._check_event(metric['console_event']):
                print_list.append(metric_name)

            if self.tb_writer is not None and self._check_event(metric['tb_event']):
                global_step = getattr(self, metric['tb_global_step'])
                self.tb_writer.add_scalar(metric['tb_name'], self.get_metric_value(metric_name),
                                          global_step=global_step)

        if self.console_logger is not None and print_list:
            print_list.append(self._speed_metric)

            log_str = f'[{self.n_periods:06d}] '
            for metric_name in print_list:
                metric = self.metrics[metric_name]
                metric_value = self.get_metric_value(metric_name)
                log_str += metric['console_name'] + ': ' + metric['console_format'].format(metric_value) + ' '

            self.console_logger.info(log_str)

    def _check_event(self, event, ignore_relaxed=False):
        if not ignore_relaxed and event.get('relaxed', True):
            return False

        triggered = False
        event_period = event['period']
        if isinstance(event_period, dict):
            pstates = event.get('period_states', None)
            if pstates is None:
                sorted_periods = list(sorted(event['period'].items()))
                pstates = {'sorted_periods': sorted_periods, 'p_indx': 0}
                event['period_states'] = pstates

            sorted_periods = pstates['sorted_periods']
            p_indx = pstates['p_indx']
            while p_indx + 1 < len(sorted_periods) and self.n_periods >= sorted_periods[p_indx + 1][0]:
                p_indx += 1
                pstates['p_indx'] = p_indx
                event['last_step'] = self.period_length * (sorted_periods[p_indx][0] - sorted_periods[p_indx][1])
            event_period = sorted_periods[p_indx][1]

        if event_period == 0:
            triggered = True
        elif event_period > 0:
            event_period_ticks = self.period_length * event_period
            last_step = event['last_step']
            k = (self.total_ticks - last_step) / event_period_ticks
            if k >= 1.0 or k < 0:
                event['last_step'] = last_step + math.floor(k) * event_period_ticks
                prev_k = (self._prev_total_ticks - last_step) / event_period_ticks
                triggered = k >= 1.0 and prev_k < 1.0

        if triggered:
            event['relaxed'] = True
        return triggered

    def _update_time(self, step_size):
        current_time = time.time()
        if self.last_time is None:
            delta_time = 0
        else:
            delta_time = current_time - self.last_time
        self.last_time = current_time
        if delta_time > self._max_delta_time:
            delta_time = 0

        if delta_time > 0:
            speed = step_size / delta_time
            self.add_metric_value(self._speed_metric, speed)


def parse_counter(counter_desc):
    if counter_desc is not None:
        smoothing_name, smoothing_period = counter_desc.split('_')
        smoothing_period = int(smoothing_period)

        if smoothing_name == 'ma':
            counter = MovingAverage(smoothing_period)
        else:
            raise NotImplementedError
    else:
        counter = SimpleLastValue()

    return counter


class MovingAverage(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.window = deque()
        self.sum = 0
        self.cnt = 0

    def add(self, value):
        if len(self.window) < self.window_size:
            self.sum += value
            self.window.append(value)
        else:
            first = self.window.popleft()
            self.sum -= first
            self.sum += value
            self.window.append(value)
        self.cnt += 1

    @property
    def value(self):
        if self.window:
            return self.sum / len(self.window)
        else:
            return 0

    def __len__(self):
        return self.cnt


class SimpleLastValue(object):
    def __init__(self):
        self.last_value = 0
        self.cnt = 0

    def add(self, value):
        self.last_value = value
        self.cnt += 1

    @property
    def value(self):
        return self.last_value
