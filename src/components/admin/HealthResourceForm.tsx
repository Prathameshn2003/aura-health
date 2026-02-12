import { useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { Loader2, ArrowLeft } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface HealthResource {
  id: string;
  title: string;
  category: string;
  description: string | null;
  external_link: string | null;
  status: string;
  is_active: boolean;
}

interface HealthResourceFormProps {
  resource: HealthResource | null;
  onSuccess: () => void;
  onCancel: () => void;
}

export const HealthResourceForm = ({ resource, onSuccess, onCancel }: HealthResourceFormProps) => {
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    title: resource?.title || "",
    category: resource?.category || "General Wellness",
    description: resource?.description || "",
    external_link: resource?.external_link || "",
    status: resource?.status || "Draft",
  });
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    const data = {
      title: formData.title,
      category: formData.category,
      description: formData.description || null,
      external_link: formData.external_link || null,
      status: formData.status,
      is_active: true,
    };

    let error;
    if (resource) {
      const result = await supabase
        .from('health_resources')
        .update(data)
        .eq('id', resource.id);
      error = result.error;
    } else {
      const result = await supabase
        .from('health_resources')
        .insert([data]);
      error = result.error;
    }

    if (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive",
      });
    } else {
      toast({
        title: resource ? "Updated" : "Created",
        description: `Health resource has been ${resource ? "updated" : "created"} successfully`,
      });
      onSuccess();
    }
    setLoading(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" onClick={onCancel}>
          <ArrowLeft className="w-4 h-4" />
        </Button>
        <h2 className="font-heading text-xl font-semibold text-foreground">
          {resource ? "Edit Health Resource" : "Add Health Resource"}
        </h2>
      </div>

      <form onSubmit={handleSubmit} className="glass-card rounded-xl p-6 space-y-6 max-w-2xl">
        <div className="space-y-2">
          <Label htmlFor="title">Title *</Label>
          <Input
            id="title"
            value={formData.title}
            onChange={(e) => setFormData({ ...formData, title: e.target.value })}
            required
            placeholder="Enter resource title"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="category">Category *</Label>
            <Select
              value={formData.category}
              onValueChange={(value) => setFormData({ ...formData, category: value })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select category" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Menstrual">Menstrual</SelectItem>
                <SelectItem value="PCOS">PCOS</SelectItem>
                <SelectItem value="Menopause">Menopause</SelectItem>
                <SelectItem value="General Wellness">General Wellness</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="status">Status *</Label>
            <Select
              value={formData.status}
              onValueChange={(value) => setFormData({ ...formData, status: value })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Draft">Draft</SelectItem>
                <SelectItem value="Published">Published</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="description">Description</Label>
          <Textarea
            id="description"
            value={formData.description}
            onChange={(e) => setFormData({ ...formData, description: e.target.value })}
            placeholder="Enter resource description"
            rows={5}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="external_link">External Reference Link (optional)</Label>
          <Input
            id="external_link"
            type="url"
            value={formData.external_link}
            onChange={(e) => setFormData({ ...formData, external_link: e.target.value })}
            placeholder="https://example.com/resource"
          />
        </div>

        <div className="flex gap-4 pt-4">
          <Button type="button" variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button type="submit" disabled={loading}>
            {loading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            {resource ? "Update Resource" : "Create Resource"}
          </Button>
        </div>
      </form>
    </div>
  );
};
