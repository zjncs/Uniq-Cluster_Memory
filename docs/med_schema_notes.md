# Med-LongMem 事件 Schema 与医学合理区间草稿

## 一、核心属性类别与医学合理区间

### 1. 检验指标类（update_policy: unique）
| attribute           | unit      | 正常范围          | 冲突变化幅度建议         |
|---------------------|-----------|-------------------|--------------------------|
| blood_glucose       | mmol/L    | 3.9 – 6.1         | ±1.5 ~ ±4.0（病理范围）  |
| blood_pressure_sys  | mmHg      | 90 – 140          | ±10 ~ ±30                |
| blood_pressure_dia  | mmHg      | 60 – 90           | ±5 ~ ±20                 |
| heart_rate          | bpm       | 60 – 100          | ±10 ~ ±30                |
| body_temperature    | ℃         | 36.1 – 37.2       | ±0.5 ~ ±1.5              |
| hemoglobin          | g/L       | 120 – 160         | ±10 ~ ±30                |
| creatinine          | μmol/L    | 44 – 133          | ±20 ~ ±60                |

### 2. 诊断类（update_policy: unique，time_scope: global）
| attribute           | 值示例                                  |
|---------------------|-----------------------------------------|
| primary_diagnosis   | 2型糖尿病, 高血压, 冠心病, 慢性肾病     |
| comorbidity         | 高脂血症, 脂肪肝, 甲状腺功能减退        |

### 3. 用药类（update_policy: latest，time_scope: global）
| attribute           | 值示例                                  |
|---------------------|-----------------------------------------|
| medication          | 二甲双胍 0.5g bid, 阿托伐他汀 20mg qn  |
| medication_allergy  | 青霉素过敏, 磺胺类过敏                  |

### 4. 症状类（update_policy: append）
| attribute           | 值示例                                  |
|---------------------|-----------------------------------------|
| symptom             | 头晕, 多尿, 视物模糊, 下肢水肿          |
| chief_complaint     | 反复头痛3个月, 血糖控制不佳             |

## 二、事件 schema（RawEvent）
```
RawEvent:
  event_id: str              # e.g., "evt_001"
  dialogue_id: str
  turn_id: int               # 在对话中的轮次位置
  speaker: "patient" | "doctor"
  attribute: str             # 对应 CanonicalMemory.attribute
  value: str
  unit: str
  time_scope: str            # 事件发生的时间范围
  event_type: str            # "measurement" | "diagnosis" | "medication" | "symptom"
  adversarial_tag: str | null  # null | "duplicate" | "conflict" | "coref" | "update"
  coref_target: str | null   # 若为 coref，指向被指代的 event_id
```

## 三、对抗注入类型定义
1. duplicate：同一 (attribute, time_scope, value) 在不同轮次重复出现
2. conflict：同一 (attribute, time_scope) 出现不同 value（在医学合理范围内变化）
3. coref：用"那个指标"/"上次的结果"等表达指代前文某个 event
4. update：用药/诊断发生覆盖性更新（latest 策略触发）

## 四、难度分级规则
- Easy：无 conflict，coref 跨度 ≤ 5 轮，事件数 5–7
- Medium：1 个 conflict，coref 跨度 5–15 轮，事件数 8–10
- Hard：≥ 2 个 conflict，coref 跨度 ≥ 15 轮，事件数 10–12，含 update
