import { useState } from "react";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { AdminSidebar } from "@/components/admin/AdminSidebar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "@/hooks/use-toast";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Loader2, Plus, Pencil, Trash2, Stethoscope, MapPin, Phone, Mail, X
} from "lucide-react";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle
} from "@/components/ui/dialog";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue
} from "@/components/ui/select";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow
} from "@/components/ui/table";

interface Doctor {
  id: string;
  name: string;
  specialization: string;
  hospital: string | null;
  location: string | null;
  phone: string | null;
  email: string | null;
  description: string | null;
  image_url: string | null;
  is_active: boolean | null;
}

const SPECIALIZATIONS = [
  "Gynecologist",
  "Obstetrician",
  "Fertility Specialist",
  "Endocrinologist",
  "General Physician",
  "Dermatologist",
  "Nutritionist",
  "Other",
];

const emptyForm = {
  name: "",
  specialization: "Gynecologist",
  hospital: "",
  location: "",
  phone: "",
  email: "",
  description: "",
};

const AdminDoctorsPage = () => {
  const queryClient = useQueryClient();
  const [formOpen, setFormOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState(emptyForm);
  const [deleteTarget, setDeleteTarget] = useState<Doctor | null>(null);

  const { data: doctors = [], isLoading } = useQuery({
    queryKey: ["admin-doctors"],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("doctors")
        .select("*")
        .order("created_at", { ascending: false });
      if (error) throw error;
      return data as Doctor[];
    },
  });

  const saveMutation = useMutation({
    mutationFn: async () => {
      if (!form.name || !form.specialization) throw new Error("Name and specialization required");
      const payload = {
        name: form.name,
        specialization: form.specialization,
        hospital: form.hospital || null,
        location: form.location || null,
        phone: form.phone || null,
        email: form.email || null,
        description: form.description || null,
      };
      if (editingId) {
        const { error } = await supabase.from("doctors").update(payload).eq("id", editingId);
        if (error) throw error;
      } else {
        const { error } = await supabase.from("doctors").insert(payload);
        if (error) throw error;
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-doctors"] });
      toast({ title: editingId ? "Doctor updated" : "Doctor added", description: "Saved successfully." });
      closeForm();
    },
    onError: (err: any) => {
      toast({ title: "Error", description: err.message || "Failed to save.", variant: "destructive" });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      const { error } = await supabase.from("doctors").delete().eq("id", id);
      if (error) throw error;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-doctors"] });
      toast({ title: "Doctor deleted", description: "Record removed successfully." });
      setDeleteTarget(null);
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to delete doctor.", variant: "destructive" });
    },
  });

  const openAdd = () => {
    setEditingId(null);
    setForm(emptyForm);
    setFormOpen(true);
  };

  const openEdit = (doc: Doctor) => {
    setEditingId(doc.id);
    setForm({
      name: doc.name,
      specialization: doc.specialization,
      hospital: doc.hospital || "",
      location: doc.location || "",
      phone: doc.phone || "",
      email: doc.email || "",
      description: doc.description || "",
    });
    setFormOpen(true);
  };

  const closeForm = () => {
    setFormOpen(false);
    setEditingId(null);
    setForm(emptyForm);
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="flex pt-16 sm:pt-20">
        <AdminSidebar />
        <main className="flex-1 p-4 md:p-8 lg:ml-0">
          <div className="max-w-6xl mx-auto space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="font-heading text-2xl md:text-3xl font-bold text-foreground flex items-center gap-3">
                  <Stethoscope className="w-7 h-7 text-teal" />
                  Manage Doctors
                </h1>
                <p className="text-muted-foreground mt-1">Add, edit, and manage doctor records</p>
              </div>
              <Button onClick={openAdd} className="gap-2">
                <Plus className="w-4 h-4" /> Add Doctor
              </Button>
            </div>

            {isLoading ? (
              <div className="flex justify-center py-12">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
              </div>
            ) : doctors.length === 0 ? (
              <div className="glass-card rounded-xl p-8 text-center">
                <Stethoscope className="w-12 h-12 text-muted-foreground mx-auto mb-3" />
                <p className="text-muted-foreground">No doctors added yet.</p>
              </div>
            ) : (
              <div className="glass-card rounded-xl overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Specialization</TableHead>
                      <TableHead className="hidden md:table-cell">Hospital</TableHead>
                      <TableHead className="hidden lg:table-cell">Location</TableHead>
                      <TableHead className="hidden lg:table-cell">Contact</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {doctors.map((doc) => (
                      <TableRow key={doc.id}>
                        <TableCell className="font-medium">{doc.name}</TableCell>
                        <TableCell>{doc.specialization}</TableCell>
                        <TableCell className="hidden md:table-cell">{doc.hospital || "—"}</TableCell>
                        <TableCell className="hidden lg:table-cell">{doc.location || "—"}</TableCell>
                        <TableCell className="hidden lg:table-cell">
                          <div className="text-xs space-y-0.5">
                            {doc.phone && <div className="flex items-center gap-1"><Phone className="w-3 h-3" />{doc.phone}</div>}
                            {doc.email && <div className="flex items-center gap-1"><Mail className="w-3 h-3" />{doc.email}</div>}
                          </div>
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            <Button size="sm" variant="outline" onClick={() => openEdit(doc)}>
                              <Pencil className="w-3.5 h-3.5" />
                            </Button>
                            <Button size="sm" variant="ghost" className="text-destructive hover:bg-destructive/10" onClick={() => setDeleteTarget(doc)}>
                              <Trash2 className="w-3.5 h-3.5" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </div>
        </main>
      </div>
      <Footer />

      {/* Add/Edit Dialog */}
      <Dialog open={formOpen} onOpenChange={closeForm}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>{editingId ? "Edit Doctor" : "Add New Doctor"}</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 mt-2">
            <div>
              <label className="text-sm font-medium text-foreground">Name *</label>
              <Input value={form.name} onChange={(e) => setForm(f => ({ ...f, name: e.target.value }))} placeholder="Dr. Full Name" />
            </div>
            <div>
              <label className="text-sm font-medium text-foreground">Specialization *</label>
              <Select value={form.specialization} onValueChange={(v) => setForm(f => ({ ...f, specialization: v }))}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {SPECIALIZATIONS.map((s) => <SelectItem key={s} value={s}>{s}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="text-sm font-medium text-foreground">Hospital / Clinic</label>
              <Input value={form.hospital} onChange={(e) => setForm(f => ({ ...f, hospital: e.target.value }))} placeholder="Hospital name" />
            </div>
            <div>
              <label className="text-sm font-medium text-foreground">Location</label>
              <Input value={form.location} onChange={(e) => setForm(f => ({ ...f, location: e.target.value }))} placeholder="City, Area" />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-sm font-medium text-foreground">Phone</label>
                <Input value={form.phone} onChange={(e) => setForm(f => ({ ...f, phone: e.target.value }))} placeholder="+91..." />
              </div>
              <div>
                <label className="text-sm font-medium text-foreground">Email</label>
                <Input value={form.email} onChange={(e) => setForm(f => ({ ...f, email: e.target.value }))} placeholder="doctor@email.com" />
              </div>
            </div>
            <div>
              <label className="text-sm font-medium text-foreground">Description</label>
              <Textarea value={form.description} onChange={(e) => setForm(f => ({ ...f, description: e.target.value }))} placeholder="Brief description..." rows={3} />
            </div>
            <div className="flex justify-end gap-3 pt-2">
              <Button variant="outline" onClick={closeForm}>Cancel</Button>
              <Button onClick={() => saveMutation.mutate()} disabled={saveMutation.isPending}>
                {saveMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : null}
                {editingId ? "Update" : "Add"} Doctor
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteTarget} onOpenChange={() => setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Doctor?</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete <strong>{deleteTarget?.name}</strong>? This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => deleteTarget && deleteMutation.mutate(deleteTarget.id)}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleteMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default AdminDoctorsPage;
