export type ProfileRecord = {
  profile_id: string;
  unit_patient_information: {
    unit_block?: string;
    resident_name?: string;
    age: number;
    living_alone_flag: boolean;
    mobility_status: string;
    caregiver_available: boolean;
  };
  medical_history: {
    cardiac_risk_flag: boolean;
    fall_risk_flag: boolean;
    diabetes_flag: boolean;
    dementia_risk_flag: boolean;
    recent_discharge_flag: boolean;
  };
  historical_call_history: {
    calls_last_7d: number;
    calls_last_30d: number;
    false_alarm_rate: number;
    time_since_last_call: number;
    average_call_duration: number;
  };
};

export type IntakeArtifactInput = {
  name: string;
  file_type: string;
  notes?: string;
};

export type OperatorAction =
  | "operator_callback"
  | "community_response"
  | "ambulance_dispatch";

export type CaseIntakePayload = {
  profile_id?: string;
  custom_profile?: CustomProfileInput;
  intake_artifacts: IntakeArtifactInput[];
};

export type CustomProfileInput = {
  profile_id?: string;
  unit_patient_information: {
    unit_block?: string;
    resident_name?: string;
    age: number;
    living_alone_flag: boolean;
    mobility_status: string;
    caregiver_available: boolean;
  };
  medical_history: {
    cardiac_risk_flag: boolean;
    fall_risk_flag: boolean;
    diabetes_flag: boolean;
    dementia_risk_flag: boolean;
    recent_discharge_flag: boolean;
  };
  historical_call_history: {
    calls_last_7d: number;
    calls_last_30d: number;
    false_alarm_rate: number;
    time_since_last_call: number;
    average_call_duration: number;
  };
};

export type CaseRecord = {
  case_id: string;
  profile_id: string;
  status: "unprocessed" | "processed" | "operator_processed";
  emergency_type?: string;
  distress_level?: "high" | "medium" | "low";
  confidence?: number;
  recommended_action?: string;
  operator_action?: OperatorAction;
  top_contributing_reasons?: string[];
  score_result?: {
    score: number;
    recommended_priority: 1 | 2 | 3;
    recommended_action: OperatorAction;
    confidence: number;
    factors: Array<{
      key: string;
      evidence: string;
      direction: "risk_up" | "risk_down";
      weight: number;
      source_module: string;
    }>;
  };
  audio_module?: {
    speech_cues?: string[];
    non_speech_cues?: string[];
    speech_distress_score?: number;
    non_speech_distress_score?: number;
    estimated_emergency_type?: string;
  };
  created_at: string;
  last_updated_at?: string;
};
