import { useState, useRef, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { ShieldCheck, Upload, CheckCircle2, AlertCircle, AlertTriangle, ScanLine } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { verifyLabel, type VerificationResponse } from "@/lib/api";

const formSchema = z.object({
  brand_name: z.string().min(1, "Brand name is required"),
  product_type: z.string().min(1, "Product type is required"),
  alcohol_content: z.string().min(1, "Alcohol content is required"),
  net_contents: z.string().optional(),
  image: z.any()
    .refine((files) => files?.length === 1, "Label image is required")
});

type FormValues = z.infer<typeof formSchema>;

export default function VerificationPage() {
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [verificationResult, setVerificationResult] = useState<VerificationResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [formEntries, setFormEntries] = useState<FormValues | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [scale, setScale] = useState(1);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const { register, handleSubmit, formState: { errors }, watch, setValue, reset } = useForm<FormValues>({
    resolver: zodResolver(formSchema),
  });
  
  // Register the image input separately to handle ref properly
  const imageRegister = register("image");

  const imageFile = watch("image");

  useEffect(() => {
    if (imageFile && imageFile.length > 0) {
      const file = imageFile[0];
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, [imageFile]);

  // Cleanup progress interval on unmount
  useEffect(() => {
    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
    };
  }, []);

  // Calculate scale for highlights when image resizes
  useEffect(() => {
    const updateScale = () => {
      if (imageRef.current && verificationResult) {
        const renderedWidth = imageRef.current.clientWidth;
        const naturalWidth = verificationResult.image_dimensions.width;
        setScale(renderedWidth / naturalWidth);
      }
    };

    window.addEventListener("resize", updateScale);
    setTimeout(updateScale, 100);
    
    return () => window.removeEventListener("resize", updateScale);
  }, [verificationResult, imagePreview]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // First, call react-hook-form's onChange to register the file
    // This must happen first so the form state is updated
    if (imageRegister.onChange) {
      imageRegister.onChange(e);
    }
    
    // Then create preview
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      // Update the file input so react-hook-form can see it
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      if (fileInputRef.current) {
        fileInputRef.current.files = dataTransfer.files;
        
        // Create preview
        const reader = new FileReader();
        reader.onloadend = () => {
          setImagePreview(reader.result as string);
        };
        reader.readAsDataURL(file);
        
        // Create a proper ChangeEvent to trigger react-hook-form
        const changeEvent = new Event("change", { bubbles: true }) as any;
        Object.defineProperty(changeEvent, "target", {
          value: fileInputRef.current,
          enumerable: true,
        });
        
        // Call handleFileChange which will call imageRegister.onChange
        handleFileChange(changeEvent as React.ChangeEvent<HTMLInputElement>);
      }
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const onSubmit = async (data: FormValues) => {
    // Clear old results immediately when starting new verification
    setVerificationResult(null);
    setFormEntries(null);
    
    setIsLoading(true);
    setProgress(0);
    
    // Clear any existing interval
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
    }
    
    // Simulate progress updates (will be replaced with actual progress later)
    // Progress increases gradually, with faster updates at the end
    let currentProgress = 0;
    progressIntervalRef.current = setInterval(() => {
      if (currentProgress < 90) {
        // Slow progress at start (OCR takes time)
        currentProgress += Math.random() * 3 + 1;
        if (currentProgress > 90) currentProgress = 90;
        setProgress(Math.min(currentProgress, 90));
      }
    }, 200);
    
    try {
      // Get the file from the input directly if form data doesn't have it
      const imageFile = data.image?.[0] || fileInputRef.current?.files?.[0];
      
      if (!imageFile) {
        alert("Please select an image file");
        setIsLoading(false);
        setProgress(0);
        if (progressIntervalRef.current) {
          clearInterval(progressIntervalRef.current);
        }
        return;
      }
      
      // Normalize form inputs: trim spaces and handle case
      const normalizedBrandName = data.brand_name.trim();
      const normalizedProductType = data.product_type.trim();
      const normalizedAlcoholContent = data.alcohol_content.trim();
      const normalizedNetContents = data.net_contents?.trim() || "";
      
      // Store form entries for display
      setFormEntries({
        brand_name: normalizedBrandName,
        product_type: normalizedProductType,
        alcohol_content: normalizedAlcoholContent,
        net_contents: normalizedNetContents || undefined,
        image: data.image,
      });
      
      const formData = new FormData();
      formData.append("brand_name", normalizedBrandName);
      formData.append("product_type", normalizedProductType);
      formData.append("alcohol_content", normalizedAlcoholContent);
      if (normalizedNetContents) {
        formData.append("net_contents", normalizedNetContents);
      }
      formData.append("image", imageFile);

      setProgress(95); // Almost done
      const result = await verifyLabel(formData);
      
      // Clear progress interval
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
      setProgress(100);
      
      setVerificationResult(result);
      
      setTimeout(() => {
        if (imageRef.current) {
          const renderedWidth = imageRef.current.clientWidth;
          const naturalWidth = result.image_dimensions.width;
          setScale(renderedWidth / naturalWidth);
        }
      }, 100);
      
      // Clear the form after successful processing (but keep formEntries for display)
      resetForm();
      // Note: formEntries is kept to show what was entered for comparison
    } catch (error) {
      console.error("Verification error:", error);
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
      setProgress(0);
    } finally {
      setIsLoading(false);
      // Reset progress after a short delay to show 100%
      setTimeout(() => {
        setProgress(0);
      }, 500);
    }
  };

  const resetForm = () => {
    // Reset react-hook-form (clear all input fields)
    reset({
      brand_name: "",
      product_type: "",
      alcohol_content: "",
      net_contents: "",
      image: undefined,
    });
    
    // Clear file input (but keep image preview visible for results)
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    
    // Note: We don't clear imagePreview here because it's needed
    // to display the image with highlights in the results panel
  };

  return (
    <div className="min-h-screen flex overflow-hidden">
      {/* Left Panel: Input and Controls */}
      <div className="w-[400px] bg-[#f5f5f5] flex flex-col p-6 border-r border-[#e5e5e5] overflow-y-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-2">
            <ShieldCheck className="w-6 h-6 text-[#2d2d2d]" />
            <h1 className="text-2xl font-bold text-[#1a1a1a] tracking-tight">
              TTB VERIFIER
            </h1>
          </div>
          <p className="text-xs text-[#6b6b6b] font-medium">
            AI-POWERED LABEL COMPLIANCE SYSTEM v1.0
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit(onSubmit)} className="flex-1 flex flex-col space-y-6">
          <div className="space-y-4">
            {/* Brand Name */}
            <div className="space-y-2">
              <Label htmlFor="brand_name" className="text-xs font-semibold text-[#1a1a1a] uppercase tracking-wide">
                BRAND NAME
              </Label>
              <Input
                id="brand_name"
                {...register("brand_name")}
                className="bg-white border-[#e5e5e5] rounded-none h-10"
                placeholder="e.g. OLD TOM DISTILLERY"
              />
              {errors.brand_name && <span className="text-red-500 text-xs">{errors.brand_name.message}</span>}
            </div>

            {/* Product Class/Type */}
            <div className="space-y-2">
              <Label htmlFor="product_type" className="text-xs font-semibold text-[#1a1a1a] uppercase tracking-wide">
                PRODUCT CLASS/TYPE
              </Label>
              <Input
                id="product_type"
                {...register("product_type")}
                className="bg-white border-[#e5e5e5] rounded-none h-10"
                placeholder="e.g. BOURBON WHISKEY"
              />
              {errors.product_type && <span className="text-red-500 text-xs">{errors.product_type.message}</span>}
            </div>

            {/* Alc. Content and Net Contents */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="alcohol_content" className="text-xs font-semibold text-[#1a1a1a] uppercase tracking-wide">
                  ALC. CONTENT
                </Label>
                <Input
                  id="alcohol_content"
                  {...register("alcohol_content")}
                  className="bg-white border-[#e5e5e5] rounded-none h-10"
                  placeholder="e.g. 45%"
                />
                {errors.alcohol_content && <span className="text-red-500 text-xs">{errors.alcohol_content.message}</span>}
              </div>

              <div className="space-y-2">
                <Label htmlFor="net_contents" className="text-xs font-semibold text-[#1a1a1a] uppercase tracking-wide">
                  NET CONTENTS
                </Label>
                <Input
                  id="net_contents"
                  {...register("net_contents")}
                  className="bg-white border-[#e5e5e5] rounded-none h-10"
                  placeholder="e.g. 750 ML"
                />
              </div>
            </div>

            {/* Label Image Upload */}
            <div className="space-y-2 pt-2">
              <Label htmlFor="image" className="text-xs font-semibold text-[#1a1a1a] uppercase tracking-wide">
                LABEL IMAGE
              </Label>
              <label
                htmlFor="image"
                className="border-2 border-dashed border-[#d0d0d0] bg-white p-8 text-center cursor-pointer hover:border-[#2d2d2d] transition-colors block"
                onDrop={handleDrop}
                onDragOver={handleDragOver}
              >
                <input
                  type="file"
                  id="image"
                  accept="image/*"
                  className="hidden"
                  name={imageRegister.name}
                  onChange={handleFileChange}
                  onBlur={imageRegister.onBlur}
                  ref={(e) => {
                    fileInputRef.current = e;
                    // Register ref with react-hook-form
                    if (typeof imageRegister.ref === "function") {
                      imageRegister.ref(e);
                    } else if (imageRegister.ref) {
                      (imageRegister.ref as React.MutableRefObject<HTMLInputElement | null>).current = e;
                    }
                  }}
                />
                <div className="flex flex-col items-center gap-2 text-[#6b6b6b] pointer-events-none">
                  <Upload className="w-8 h-8" />
                  <span className="text-xs font-medium uppercase">
                    DROP LABEL OR CLICK TO UPLOAD
                  </span>
                </div>
              </label>
              {errors.image && <span className="text-red-500 text-xs">{String(errors.image.message)}</span>}
            </div>
          </div>

          {/* Verify Compliance Button with Progress Bar */}
          <Button
            type="submit"
            className="w-full bg-[#2d2d2d] text-white hover:bg-[#1a1a1a] rounded-none h-12 text-base font-semibold uppercase tracking-wider mt-auto relative overflow-hidden"
            disabled={isLoading}
          >
            {isLoading ? (
              <span className="flex items-center gap-2 relative z-10">
                <ScanLine className="animate-pulse" /> Processing... {Math.round(progress)}%
              </span>
            ) : (
              "VERIFY COMPLIANCE"
            )}
            {isLoading && (
              <div
                className="absolute inset-0 bg-green-600 transition-all duration-300 ease-out"
                style={{ width: `${progress}%` }}
              />
            )}
          </Button>
        </form>
      </div>

      {/* Right Panel: Display Area */}
      <div className="flex-1 bg-white relative overflow-hidden">
        {/* Grid Background Pattern */}
        <div
          className="absolute inset-0 pointer-events-none opacity-[0.03]"
          style={{
            backgroundImage:
              "linear-gradient(#000 1px, transparent 1px), linear-gradient(90deg, #000 1px, transparent 1px)",
            backgroundSize: "20px 20px",
          }}
        />

        {/* Display Area */}
        <div className="absolute inset-0 flex items-center justify-center p-6">
          <div className="w-full h-full max-w-4xl max-h-[600px] border border-[#1a1a1a] bg-white relative">
            {!imagePreview ? (
              <div className="w-full h-full relative flex items-center justify-center">
                {/* Scanning bracket icon in top right */}
                <div className="absolute top-4 right-4 w-8 h-8 text-[#9b9b9b]">
                  <svg
                    viewBox="0 0 32 32"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    className="w-full h-full"
                  >
                    <path
                      d="M8 8 L8 12 L12 12 L12 8 Z M20 8 L20 12 L24 12 L24 8 Z M8 20 L8 24 L12 24 L12 20 Z M20 20 L20 24 L24 24 L24 20 Z"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      fill="none"
                    />
                  </svg>
                </div>
                {/* Centered message */}
                <p className="text-sm font-medium text-[#9b9b9b] uppercase tracking-widest">
                  WAITING FOR INPUT SIGNAL...
                </p>
              </div>
            ) : (
              <div className="relative w-full h-full p-4 flex items-center justify-center bg-neutral-100/50">
                <div className="relative inline-block max-w-full max-h-full shadow-xl">
                  <img 
                    ref={imageRef}
                    src={imagePreview} 
                    alt="Label Preview" 
                    className="max-w-full max-h-[60vh] object-contain border border-[#e5e5e5]"
                  />
                  
                  {/* OCR Highlights Overlay */}
                  {verificationResult && verificationResult.highlights.map((highlight, idx) => (
                    <div
                      key={idx}
                      className="absolute border-2 border-green-500 bg-green-500/10"
                      style={{
                        left: `${highlight.x * scale}px`,
                        top: `${highlight.y * scale}px`,
                        width: `${highlight.w * scale}px`,
                        height: `${highlight.h * scale}px`,
                      }}
                    />
                  ))}
                  
                  {/* Scanning Effect Animation */}
                  {isLoading && (
                    <div className="absolute inset-0 bg-gradient-to-b from-transparent via-green-500/20 to-transparent h-[20%] w-full animate-[scan_2s_ease-in-out_infinite]" />
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Results Panel */}
        {verificationResult && (
          <div className="absolute bottom-6 left-6 right-6 max-w-4xl mx-auto bg-white border-2 border-[#1a1a1a] shadow-lg animate-in slide-in-from-bottom-10 duration-500 max-h-[300px] overflow-y-auto">
            <div className="flex items-center justify-between p-4 border-b border-[#e5e5e5] bg-[#f5f5f5]">
              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${verificationResult.success ? "bg-green-500" : "bg-red-500"}`} />
                <h2 className="text-lg font-bold">
                  STATUS: {verificationResult.success ? "VERIFIED" : "DISCREPANCIES FOUND"}
                </h2>
              </div>
            </div>

            <div className="p-4 space-y-3">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <ResultItem 
                  label="Brand Name" 
                  match={verificationResult.results.brand_name.match} 
                  value={verificationResult.results.brand_name.value} 
                />
                <ResultItem 
                  label="Product Class" 
                  match={verificationResult.results.product_type.match} 
                  value={verificationResult.results.product_type.value} 
                />
                <ResultItem 
                  label="Alcohol Content" 
                  match={verificationResult.results.alcohol_content.match} 
                  value={verificationResult.results.alcohol_content.value} 
                />
                {verificationResult.results.net_contents && (
                  <ResultItem 
                    label="Net Contents" 
                    match={verificationResult.results.net_contents.match} 
                    value={verificationResult.results.net_contents.value} 
                  />
                )}
              </div>
              
              <div className="pt-2 border-t border-dashed border-[#e5e5e5]">
                <div className="flex items-start gap-3">
                  {verificationResult.results.compliance.compliant ? (
                    <CheckCircle2 className="w-5 h-5 text-green-500 mt-0.5" />
                  ) : (
                    <AlertTriangle className="w-5 h-5 text-yellow-500 mt-0.5" />
                  )}
                  <div>
                    <p className="font-bold text-sm uppercase mb-1">Government Warning</p>
                    <div className="grid grid-cols-2 gap-x-8 gap-y-1 text-xs font-mono text-[#6b6b6b]">
                      <span className={verificationResult.results.compliance.details.has_warning_label ? "text-green-500" : "text-red-500"}>
                        [{verificationResult.results.compliance.details.has_warning_label ? "OK" : "MISSING"}] LABEL HEADER
                      </span>
                      <span className={verificationResult.results.compliance.details.has_surgeon_general ? "text-green-500" : "text-red-500"}>
                        [{verificationResult.results.compliance.details.has_surgeon_general ? "OK" : "MISSING"}] SURGEON GENERAL
                      </span>
                      <span className={verificationResult.results.compliance.details.has_pregnancy_warning ? "text-green-500" : "text-red-500"}>
                        [{verificationResult.results.compliance.details.has_pregnancy_warning ? "OK" : "MISSING"}] PREGNANCY
                      </span>
                      <span className={verificationResult.results.compliance.details.has_driving_warning ? "text-green-500" : "text-red-500"}>
                        [{verificationResult.results.compliance.details.has_driving_warning ? "OK" : "MISSING"}] MACHINERY
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* OCR Text Display (Collapsible) */}
              <div className="pt-2 border-t border-dashed border-[#e5e5e5]">
                <details className="group">
                  <summary className="cursor-pointer text-xs font-bold uppercase text-[#6b6b6b] hover:text-[#1a1a1a] transition-colors">
                    📄 View OCR Text (What the system read from the label)
                  </summary>
                  <div className="mt-2 p-3 bg-[#f5f5f5] border border-[#e5e5e5] rounded">
                    <p className="text-xs font-mono text-[#1a1a1a] whitespace-pre-wrap break-words max-h-40 overflow-y-auto">
                      {verificationResult.ocr_text_snippet}
                    </p>
                    <p className="text-[10px] text-[#6b6b6b] mt-2">
                      (First 500 characters of extracted text)
                    </p>
                  </div>
                </details>
              </div>
              
              {/* Form Entry Display (Collapsible) */}
              {formEntries && (
                <div className="pt-2 border-t border-dashed border-[#e5e5e5]">
                  <details className="group">
                    <summary className="cursor-pointer text-xs font-bold uppercase text-[#6b6b6b] hover:text-[#1a1a1a] transition-colors">
                      📝 Form Entry (What was manually entered)
                    </summary>
                    <div className="mt-2 p-3 bg-[#f5f5f5] border border-[#e5e5e5] rounded">
                      <div className="space-y-2 text-xs">
                        <div>
                          <span className="font-semibold text-[#1a1a1a]">Brand Name:</span>
                          <span className="ml-2 font-mono text-[#6b6b6b]">{formEntries.brand_name || "—"}</span>
                        </div>
                        <div>
                          <span className="font-semibold text-[#1a1a1a]">Product Type:</span>
                          <span className="ml-2 font-mono text-[#6b6b6b]">{formEntries.product_type || "—"}</span>
                        </div>
                        <div>
                          <span className="font-semibold text-[#1a1a1a]">Alcohol Content:</span>
                          <span className="ml-2 font-mono text-[#6b6b6b]">{formEntries.alcohol_content || "—"}</span>
                        </div>
                        {formEntries.net_contents && (
                          <div>
                            <span className="font-semibold text-[#1a1a1a]">Net Contents:</span>
                            <span className="ml-2 font-mono text-[#6b6b6b]">{formEntries.net_contents}</span>
                          </div>
                        )}
                      </div>
                      <p className="text-[10px] text-[#6b6b6b] mt-2">
                        (For manual audit and comparison)
                      </p>
                    </div>
                  </details>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ResultItem({ label, match, value }: { label: string, match: boolean, value: string | null }) {
  return (
    <div className="flex items-start gap-3 p-2 hover:bg-[#f5f5f5] transition-colors border border-transparent hover:border-[#e5e5e5]">
      {match ? (
        <CheckCircle2 className="w-5 h-5 text-green-500 mt-0.5" />
      ) : (
        <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
      )}
      <div>
        <p className="font-bold text-sm uppercase">{label}</p>
        <p className="font-mono text-xs text-[#6b6b6b]">{value || "N/A"}</p>
        {!match && <p className="text-[10px] text-red-500 font-bold mt-1">MISMATCH DETECTED</p>}
      </div>
    </div>
  );
}

