from collections import defaultdict
class Calibrate:
    pickle_kv_cache = False
    target_value_head_predicted = 0
    target_key_head_predicted = 0
    total_heads = 0
    focus_layer_sink_tokens = defaultdict(set)
    ref_layer_sink_tokens = defaultdict(set)
    change_focus = False
    predict_values = False
    predict_keys = False

    total_tokens_collected = 0

    @staticmethod
    def set_pickle_kv_cache(pickle_kv_cache):
        Calibrate.pickle_kv_cache = pickle_kv_cache

    @staticmethod
    def get_pickle_kv_cache():
        return Calibrate.pickle_kv_cache

