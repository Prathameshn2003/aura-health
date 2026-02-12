import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { AdminHealthResources } from "@/components/admin/AdminHealthResources";
import { AdminSidebar } from "@/components/admin/AdminSidebar";

const AdminResourcesPage = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="flex pt-20">
        <AdminSidebar />
        <main className="flex-1 p-6 lg:p-8">
          <AdminHealthResources />
        </main>
      </div>
      <Footer />
    </div>
  );
};

export default AdminResourcesPage;
