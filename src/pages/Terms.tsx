import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { FileText } from "lucide-react";

const Terms = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="pt-24 pb-16">
        <div className="container mx-auto px-4 max-w-4xl">
          <div className="mb-10">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 rounded-lg bg-accent/20 flex items-center justify-center">
                <FileText className="w-5 h-5 text-accent" />
              </div>
              <h1 className="font-heading text-3xl md:text-4xl font-bold text-foreground">
                Terms of Service
              </h1>
            </div>
            <p className="text-muted-foreground">
              Last updated: December 2024
            </p>
          </div>

          <div className="glass-card rounded-xl p-8 space-y-8">
            <section>
              <h2 className="font-heading text-xl font-semibold text-foreground mb-4">
                1. Acceptance of Terms
              </h2>
              <p className="text-muted-foreground">
                By accessing or using NaariCare, you agree to be bound by these Terms of Service. 
                If you do not agree to these terms, please do not use our services.
              </p>
            </section>

            <section>
              <h2 className="font-heading text-xl font-semibold text-foreground mb-4">
                2. Description of Service
              </h2>
              <p className="text-muted-foreground">
                NaariCare is a women's health platform providing health tracking, AI-powered insights, 
                educational resources, and connections to healthcare providers. Our services are for 
                informational purposes only and do not constitute medical advice.
              </p>
            </section>

            <section>
              <h2 className="font-heading text-xl font-semibold text-foreground mb-4">
                3. Medical Disclaimer
              </h2>
              <p className="text-muted-foreground">
                NaariCare is not a substitute for professional medical advice, diagnosis, or treatment. 
                Always seek the advice of your physician or other qualified health provider with any 
                questions you may have regarding a medical condition. Never disregard professional 
                medical advice or delay in seeking it because of something you have read on NaariCare.
              </p>
            </section>

            <section>
              <h2 className="font-heading text-xl font-semibold text-foreground mb-4">
                4. User Responsibilities
              </h2>
              <p className="text-muted-foreground mb-4">
                As a user, you agree to:
              </p>
              <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
                <li>Provide accurate information when creating your account</li>
                <li>Keep your login credentials secure</li>
                <li>Use the service for lawful purposes only</li>
                <li>Not share your account with others</li>
                <li>Report any security concerns immediately</li>
              </ul>
            </section>

            <section>
              <h2 className="font-heading text-xl font-semibold text-foreground mb-4">
                5. Intellectual Property
              </h2>
              <p className="text-muted-foreground">
                All content, features, and functionality of NaariCare, including but not limited to 
                text, graphics, logos, and software, are the exclusive property of NaariCare and are 
                protected by intellectual property laws.
              </p>
            </section>

            <section>
              <h2 className="font-heading text-xl font-semibold text-foreground mb-4">
                6. Limitation of Liability
              </h2>
              <p className="text-muted-foreground">
                NaariCare shall not be liable for any indirect, incidental, special, consequential, 
                or punitive damages resulting from your use of or inability to use the service.
              </p>
            </section>

            <section>
              <h2 className="font-heading text-xl font-semibold text-foreground mb-4">
                7. Changes to Terms
              </h2>
              <p className="text-muted-foreground">
                We reserve the right to modify these terms at any time. We will notify users of 
                significant changes via email or through the platform. Continued use of NaariCare 
                after changes constitutes acceptance of the new terms.
              </p>
            </section>

            <section>
              <h2 className="font-heading text-xl font-semibold text-foreground mb-4">
                8. Contact Information
              </h2>
              <p className="text-muted-foreground">
                For questions about these Terms of Service, please contact us at{" "}
                <a href="mailto:legal@naaricare.com" className="text-accent hover:underline">
                  legal@naaricare.com
                </a>
              </p>
            </section>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Terms;
