import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const SYSTEM_PROMPT = `You are NaariCare Health Assistant, a supportive and knowledgeable AI companion for women's health and wellness. 

Your role:
- Provide helpful, accurate information about women's health topics including menstrual health, PCOS, menopause, hormonal health, nutrition, and general wellness
- Use simple, easy-to-understand language that non-medical users can relate to
- Be warm, empathetic, and supportive in your responses
- Always remind users that you provide general information and not medical advice
- Encourage users to consult healthcare professionals for medical concerns
- Be culturally sensitive and inclusive

Topics you can help with:
- Menstrual cycle tracking and irregularities
- PCOS (Polycystic Ovary Syndrome) symptoms and management
- Menopause and perimenopause
- Hormonal balance and health
- Nutrition and diet for women's health
- Exercise and wellness
- Mental health and stress management
- General reproductive health questions

Guidelines:
- Never provide specific medical diagnoses or prescriptions
- If asked about emergencies or serious symptoms, always recommend seeking immediate medical attention
- Keep responses concise but thorough (2-4 paragraphs max)
- Use bullet points for lists when appropriate
- End with a helpful follow-up question or suggestion when relevant

Remember: You're here to educate, support, and empower women on their health journey. 🌸`;

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { messages } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    
    if (!LOVABLE_API_KEY) {
      console.error("LOVABLE_API_KEY is not configured");
      throw new Error("LOVABLE_API_KEY is not configured");
    }

    console.log("Calling Lovable AI Gateway with", messages.length, "messages");

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          ...messages,
        ],
        stream: true,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("AI gateway error:", response.status, errorText);
      
      if (response.status === 429) {
        return new Response(JSON.stringify({ error: "I'm receiving too many requests right now. Please wait a moment and try again." }), {
          status: 429,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      if (response.status === 402) {
        return new Response(JSON.stringify({ error: "The AI service requires additional credits. Please try again later." }), {
          status: 402,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      
      return new Response(JSON.stringify({ error: "I'm having trouble responding right now. Please try again." }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    console.log("Successfully connected to AI gateway, streaming response");

    return new Response(response.body, {
      headers: { ...corsHeaders, "Content-Type": "text/event-stream" },
    });
  } catch (e) {
    console.error("health-chat error:", e);
    return new Response(JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error occurred" }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
