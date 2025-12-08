import axios from "axios";

export interface Highlight {
  x: number;
  y: number;
  w: number;
  h: number;
  text?: string;
}

export interface VerificationResponse {
  success: boolean;
  ocr_text_snippet: string;
  results: {
    brand_name: { match: boolean; value: string };
    product_type: { match: boolean; value: string };
    alcohol_content: { match: boolean; value: string };
    net_contents?: { match: boolean; value: string | null };
    compliance: {
      compliant: boolean;
      details: {
        has_warning_label: boolean;
        has_surgeon_general: boolean;
        has_pregnancy_warning: boolean;
        has_driving_warning: boolean;
      };
    };
  };
  highlights: Highlight[];
  image_dimensions: { width: number; height: number };
}

export async function verifyLabel(formData: FormData): Promise<VerificationResponse> {
  const response = await axios.post<VerificationResponse>(
    "/api/verify",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );
  return response.data;
}

