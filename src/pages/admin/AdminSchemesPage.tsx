import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { AdminSchemes } from "@/components/admin/AdminSchemes";
import { AdminSidebar } from "@/components/admin/AdminSidebar";

const AdminSchemesPage = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="flex pt-20">
        <AdminSidebar />
        <main className="flex-1 p-6 lg:p-8">
          <AdminSchemes />
        </main>
      </div>
      <Footer />
    </div>
  );
};

export default AdminSchemesPage;
