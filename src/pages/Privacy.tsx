import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Shield } from "lucide-react";

const Privacy = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="pt-24 pb-16">
        <div className="container mx-auto px-4 max-w-4xl">
          <div className="mb-10">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center">
                <Shield className="w-5 h-5 text-primary" />
              </div>
              <h1 className="font-heading text-3xl md:text-4xl font-bold text-foreground">
                Privacy Policy
              </h1>
            </div>
            <p className="text-muted-foreground">
              Last updated: December 2024
            </p>
          </div>

          <div className="glass-card rounded-xl p-8 space-y-8">
            <section>
              <h2 className="font-heading text-xl font-semibold text-foreground mb-4">
                1. Information We Collect
              </h2>
              <p className="text-muted-foreground mb-4">
                NaariCare collects information you provide directly to us, including:
              </p>
              <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
                <li>Account information (name, email, password)</li>
                <li>Health data you choose to log (menstrual cycle, symptoms)</li>
                <li>Assessment responses for health predictions</li>
                <li>Usage data and preferences</li>
              </ul>
            </section>

            <section>
              <h2 className="font-heading text-xl font-semibold text-foreground mb-4">
                2. How We Use Your Information
              </h2>
              <p className="text-muted-foreground mb-4">
                We use the information we collect to:
              </p>
              <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
                <li>Provide, maintain, and improve our services</li>
                <li>Generate personalized health insights and predictions</li>
                <li>Send you updates and notifications</li>
                <li>Respond to your comments and questions</li>
                <li>Analyze usage patterns to enhance user experience</li>
              </ul>
            </section>

            <section>
              <h2 className="font-heading text-xl font-semibold text-foreground mb-4">
                3. Data Security
              </h2>
              <p className="text-muted-foreground">
                We implement industry-standard security measures to protect your personal information. 
                Your health data is encrypted both in transit and at rest. We use secure authentication 
                methods and regularly audit our security practices.
              </p>
            </section>

            <section>
              <h2 className="font-heading text-xl font-semibold text-foreground mb-4">
                4. Data Sharing
              </h2>
              <p className="text-muted-foreground">
                We do not sell your personal information. We may share data only with your consent, 
                to comply with legal obligations, or with service providers who assist in operating 
                our platform under strict confidentiality agreements.
              </p>
            </section>

            <section>
              <h2 className="font-heading text-xl font-semibold text-foreground mb-4">
                5. Your Rights
              </h2>
              <p className="text-muted-foreground mb-4">
                You have the right to:
              </p>
              <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-4">
                <li>Access your personal data</li>
                <li>Correct inaccurate information</li>
                <li>Delete your account and associated data</li>
                <li>Export your data in a portable format</li>
                <li>Opt out of non-essential communications</li>
              </ul>
            </section>

            <section>
              <h2 className="font-heading text-xl font-semibold text-foreground mb-4">
                6. Contact Us
              </h2>
              <p className="text-muted-foreground">
                If you have questions about this Privacy Policy, please contact us at{" "}
                <a href="mailto:privacy@naaricare.com" className="text-accent hover:underline">
                  privacy@naaricare.com
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

export default Privacy;
