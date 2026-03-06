import { useState } from "react";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { AdminSidebar } from "@/components/admin/AdminSidebar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "@/hooks/use-toast";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Loader2, Plus, Pencil, Trash2, Stethoscope, Phone, Mail, ShieldCheck, Sparkles
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

const SPECIALIZATIONS = [
  "Gynecologist", "Obstetrician", "PCOS Specialist", "Fertility Specialist",
  "Endocrinologist", "General Physician", "Dermatologist", "Nutritionist", "Other",
];

const emptyForm = {
  name: "", specialization: "Gynecologist", hospital: "", location: "", city: "",
  phone: "", email: "", description: "", latitude: "", longitude: "",
  experience: "", verified: false, recommended: false,
};

const AdminDoctorsPage = () => {
  const queryClient = useQueryClient();
  const [formOpen, setFormOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState(emptyForm);
  const [deleteTarget, setDeleteTarget] = useState<any>(null);

  const { data: doctors = [], isLoading } = useQuery({
    queryKey: ["admin-doctors"],
    queryFn: async () => {
      const { data, error } = await supabase.from("doctors").select("*").order("created_at", { ascending: false });
      if (error) throw error;
      return data;
    },
  });

  const saveMutation = useMutation({
    mutationFn: async () => {
      if (!form.name || !form.specialization) throw new Error("Name and specialization required");
      const payload: any = {
        name: form.name, specialization: form.specialization,
        hospital: form.hospital || null, location: form.location || null,
        city: form.city || null, phone: form.phone || null, email: form.email || null,
        description: form.description || null,
        latitude: form.latitude ? parseFloat(form.latitude) : null,
        longitude: form.longitude ? parseFloat(form.longitude) : null,
        experience: form.experience ? parseInt(form.experience) : null,
        verified: form.verified, recommended: form.recommended,
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
      toast({ title: editingId ? "Doctor updated" : "Doctor added" });
      closeForm();
    },
    onError: (err: any) => {
      toast({ title: "Error", description: err.message, variant: "destructive" });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      const { error } = await supabase.from("doctors").delete().eq("id", id);
      if (error) throw error;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-doctors"] });
      toast({ title: "Doctor deleted" });
      setDeleteTarget(null);
    },
  });

  const openEdit = (doc: any) => {
    setEditingId(doc.id);
    setForm({
      name: doc.name, specialization: doc.specialization,
      hospital: doc.hospital || "", location: doc.location || "", city: doc.city || "",
      phone: doc.phone || "", email: doc.email || "", description: doc.description || "",
      latitude: doc.latitude?.toString() || "", longitude: doc.longitude?.toString() || "",
      experience: doc.experience?.toString() || "",
      verified: doc.verified || false, recommended: doc.recommended || false,
    });
    setFormOpen(true);
  };

  const closeForm = () => { setFormOpen(false); setEditingId(null); setForm(emptyForm); };

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
                  <Stethoscope className="w-7 h-7 text-teal" /> Manage Doctors
                </h1>
                <p className="text-muted-foreground mt-1">Add, edit, and manage doctor records</p>
              </div>
              <Button onClick={() => { setEditingId(null); setForm(emptyForm); setFormOpen(true); }} className="gap-2">
                <Plus className="w-4 h-4" /> Add Doctor
              </Button>
            </div>

            {isLoading ? (
              <div className="flex justify-center py-12"><Loader2 className="w-8 h-8 animate-spin text-primary" /></div>
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
                      <TableHead className="hidden md:table-cell">City</TableHead>
                      <TableHead className="hidden lg:table-cell">Status</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {doctors.map((doc: any) => (
                      <TableRow key={doc.id}>
                        <TableCell className="font-medium">
                          {doc.name}
                          {doc.experience && <span className="text-xs text-muted-foreground ml-1">({doc.experience} yrs)</span>}
                        </TableCell>
                        <TableCell>{doc.specialization}</TableCell>
                        <TableCell className="hidden md:table-cell">{doc.city || doc.location || "—"}</TableCell>
                        <TableCell className="hidden lg:table-cell">
                          <div className="flex gap-1.5">
                            {doc.verified && <Badge variant="secondary" className="text-xs bg-teal/15 text-teal"><ShieldCheck className="w-3 h-3 mr-1" />Verified</Badge>}
                            {doc.recommended && <Badge variant="secondary" className="text-xs bg-accent/15 text-accent"><Sparkles className="w-3 h-3 mr-1" />Recommended</Badge>}
                            {!doc.is_active && <Badge variant="destructive" className="text-xs">Inactive</Badge>}
                          </div>
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            <Button size="sm" variant="outline" onClick={() => openEdit(doc)}><Pencil className="w-3.5 h-3.5" /></Button>
                            <Button size="sm" variant="ghost" className="text-destructive hover:bg-destructive/10" onClick={() => setDeleteTarget(doc)}><Trash2 className="w-3.5 h-3.5" /></Button>
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

      {/* Form Dialog */}
      <Dialog open={formOpen} onOpenChange={closeForm}>
        <DialogContent className="sm:max-w-lg max-h-[85vh] overflow-y-auto">
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
                <SelectContent>{SPECIALIZATIONS.map((s) => <SelectItem key={s} value={s}>{s}</SelectItem>)}</SelectContent>
              </Select>
            </div>
            <div>
              <label className="text-sm font-medium text-foreground">Hospital / Clinic</label>
              <Input value={form.hospital} onChange={(e) => setForm(f => ({ ...f, hospital: e.target.value }))} placeholder="Hospital name" />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-sm font-medium text-foreground">Location</label>
                <Input value={form.location} onChange={(e) => setForm(f => ({ ...f, location: e.target.value }))} placeholder="Area, City" />
              </div>
              <div>
                <label className="text-sm font-medium text-foreground">City</label>
                <Input value={form.city} onChange={(e) => setForm(f => ({ ...f, city: e.target.value }))} placeholder="Mumbai" />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-sm font-medium text-foreground">Latitude</label>
                <Input value={form.latitude} onChange={(e) => setForm(f => ({ ...f, latitude: e.target.value }))} placeholder="18.5204" type="number" step="any" />
              </div>
              <div>
                <label className="text-sm font-medium text-foreground">Longitude</label>
                <Input value={form.longitude} onChange={(e) => setForm(f => ({ ...f, longitude: e.target.value }))} placeholder="73.8567" type="number" step="any" />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-sm font-medium text-foreground">Phone</label>
                <Input value={form.phone} onChange={(e) => setForm(f => ({ ...f, phone: e.target.value }))} placeholder="+91..." />
              </div>
              <div>
                <label className="text-sm font-medium text-foreground">Experience (years)</label>
                <Input value={form.experience} onChange={(e) => setForm(f => ({ ...f, experience: e.target.value }))} placeholder="10" type="number" />
              </div>
            </div>
            <div>
              <label className="text-sm font-medium text-foreground">Email</label>
              <Input value={form.email} onChange={(e) => setForm(f => ({ ...f, email: e.target.value }))} placeholder="doctor@email.com" />
            </div>
            <div>
              <label className="text-sm font-medium text-foreground">Description</label>
              <Textarea value={form.description} onChange={(e) => setForm(f => ({ ...f, description: e.target.value }))} placeholder="Brief description..." rows={2} />
            </div>
            <div className="flex gap-6">
              <label className="flex items-center gap-2 text-sm">
                <Switch checked={form.verified} onCheckedChange={(v) => setForm(f => ({ ...f, verified: v }))} />
                Verified
              </label>
              <label className="flex items-center gap-2 text-sm">
                <Switch checked={form.recommended} onCheckedChange={(v) => setForm(f => ({ ...f, recommended: v }))} />
                Recommended
              </label>
            </div>
            <div className="flex justify-end gap-3 pt-2">
              <Button variant="outline" onClick={closeForm}>Cancel</Button>
              <Button onClick={() => saveMutation.mutate()} disabled={saveMutation.isPending}>
                {saveMutation.isPending && <Loader2 className="w-4 h-4 animate-spin mr-2" />}
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
            <AlertDialogAction onClick={() => deleteTarget && deleteMutation.mutate(deleteTarget.id)} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              {deleteMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default AdminDoctorsPage;
