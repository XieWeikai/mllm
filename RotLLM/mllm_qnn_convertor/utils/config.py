class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = ConfigDict(v)
    
    """配置字典，支持点号访问"""
    def __getattr__(self, key):
        # try:
        #     return self[key]
        # except KeyError:
        #     # 让getattr的默认值机制生效
        #     raise AttributeError(key)
        return self.get(key, None)
        
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return list(self.keys()) + list(super().__dir__())
    
    def check_schema(self, schema: dict, path: str = "<root>"):
        """
        用 JSON Schema（dict 形式）校验当前 ConfigDict。
        失败时抛 ValueError / TypeError，信息中包含字段路径。
        """
        def _err(msg):
            raise ValueError(f"{path}: {msg}")

        # 1. 类型检查
        type_req = schema.get("type")
        if type_req == "object":
            if not isinstance(self, dict):
                _err(f"expected object, got {type(self).__name__}")
        elif type_req == "array":
            if not isinstance(self, list):
                _err(f"expected array, got {type(self).__name__}")
        elif type_req in ("string", "integer", "number", "boolean"):
            if not isinstance(self, {"string": str, "integer": int,
                                    "number": (int, float),
                                    "boolean": bool}[type_req]):
                _err(f"expected {type_req}, got {type(self).__name__}")

        # 2. 对象专用检查
        if type_req == "object":
            props = schema.get("properties", {})
            required = schema.get("required", [])
            add_prop = schema.get("additionalProperties", True)

            # 2.1 必须字段
            for k in required:
                if k not in self:
                    _err(f"missing required field '{k}'")

            # 2.2 禁止额外字段
            if add_prop is False:
                extra = set(self) - set(props)
                if extra:
                    _err(f"unexpected fields {list(extra)}")

            # 2.3 递归校验每个子字段
            for k, sub_schema in props.items():
                if k in self:
                    child_path = f"{path}.{k}" if path != "<root>" else k
                    # 如果是 dict/list，递归；否则直接校验
                    if isinstance(self[k], dict):
                        ConfigDict(self[k]).check_schema(sub_schema, child_path)
                    elif isinstance(self[k], list):
                        for idx, item in enumerate(self[k]):
                            if isinstance(item, dict):
                                ConfigDict(item).check_schema(sub_schema, f"{child_path}[{idx}]")
                            else:
                                ConfigDict({"_": item}).check_schema(sub_schema, f"{child_path}[{idx}]")
                    else:
                        ConfigDict({"_": self[k]}).check_schema(sub_schema, child_path)

        # 3. 数组专用检查（简单版：元素类型递归）
        if type_req == "array":
            items_schema = schema.get("items")
            if items_schema:
                for idx, item in enumerate(self):
                    child_path = f"{path}[{idx}]"
                    if isinstance(item, dict):
                        ConfigDict(item).check_schema(items_schema, child_path)
                    else:
                        ConfigDict({"_": item}).check_schema(items_schema, child_path)

        # 4. 数值/字符串约束（示例：minimum, maximum, pattern）
        if isinstance(self, (int, float)):
            if "minimum" in schema and self < schema["minimum"]:
                _err(f"value {self} < minimum {schema['minimum']}")
            if "maximum" in schema and self > schema["maximum"]:
                _err(f"value {self} > maximum {schema['maximum']}")

        if isinstance(self, str) and "pattern" in schema:
            import re
            if not re.fullmatch(schema["pattern"], self):
                _err(f"value '{self}' does not match pattern /{schema['pattern']}/")

        # 5. 枚举值检查
        if "enum" in schema and self not in schema["enum"]:
            _err(f"value {self} not in allowed enum {schema['enum']}")
            
            
CONFIG_SCHEMA = {
    "type": "object",
    "required": ["profile_config", "export_config"],
    "additionalProperties": False,

    "properties": {
        "profile_config": {
            "type": "object",
            "required": [
                "dataset_path", "output_path", "num_samples", "no_bias", "model_config"
            ],
            "additionalProperties": False,

            "properties": {
                "dataset_path": {"type": "string"}, # which dataset to use for profiling
                "output_path":  {"type": "string"}, # where to save the profiling results
                "num_samples":  {"type": "integer", "minimum": 2}, # number of samples to use in dataset to profile
                "no_bias":      {"type": "boolean"}, # if true, we will ignore bias when profiling a linear layer. that is, for a linear layer Wx + b, we will only record the output scale of Wx.

                "model_config": {
                    "type": "object",
                    "required": [
                        "model_type", # currently only support qwen2 and qwen-vl(this is qwen2-vl, not qwen2.5-vl. you can refer to model_interface.py for details)
                        "tokenizer_name", # path to tokenizer
                        "model_name",     # path to model
                    ],
                    "additionalProperties": False,

                    "properties": {
                        "model_type":     {"type": "string"},
                        "tokenizer_name": {"type": "string"},
                        "model_name":     {"type": "string"},
                        "online_rotation": {"type": "boolean"}, # rotate after loading model
                        "random_rotate":   {"type": "boolean"}, # generate random rotation matrix and use it to rotate the model
                        "save_rotation":   {"type": "string"},  # this is the path to save the rotation matrix
                        "R_path": {"type": "string"} # if online_rotation is true, rotation matrix from R_path will be used to rotate the model. random_rotate and  R_path and random_rotate are mutually exclusive
                    }
                }
            }
        },

        "export_config": {
            "type": "object",
            "required": [
                "scale_file", "output_model", "model_config"
            ],
            "additionalProperties": False,

            "properties": {
                "scale_file":        {"type": "string"},
                "output_model":      {"type": "string"},
                "t01m_clip_threshold": {"type": "integer"},
                "quant_bias":        {"type": "boolean"},
                "clip_all":          {"type": "boolean"}, # if true, t01m_clip_threshold will not be effected

                "model_config": {
                    "type": "object",
                    "required": [
                        "model_type",
                        "tokenizer_name",
                        "model_name",
                    ],
                    "additionalProperties": False,

                    "properties": {
                        "model_type":     {"type": "string"},
                        "tokenizer_name": {"type": "string"},
                        "model_name":     {"type": "string"},
                        "online_rotation": {"type": "boolean"},
                        "random_rotate":   {"type": "boolean"},
                        "save_rotation":   {"type": "string"},
                        "R_path": {"type": "string"} # R_path and random_rotate are mutually exclusive
                    }
                }
            }
        }
    }
}

        
if __name__ == "__main__":
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age":  {"type": "integer", "minimum": 0},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "zip":    {"type": "string", "pattern": r"^\d{5}$"}
                },
                "required": ["street", "zip"],
                "additionalProperties": False
            }
        },
        "required": ["name", "address"],
        "additionalProperties": False
    }

    cfg = ConfigDict({
        "name": "Alice",
        "age": 30,
        "address": {
            "street": "Main St",
            "zip": "12345"
        }
    })

    cfg.check_schema(schema)
