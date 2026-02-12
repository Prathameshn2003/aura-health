import { useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { useToast } from "@/hooks/use-toast";
import { Loader2, ArrowLeft } from "lucide-react";

interface Scheme {
  id: string;
  name: string;
  description: string | null;
  eligibility: string | null;
  benefits: string | null;
  how_to_apply: string | null;
  website: string | null;
  category: string | null;
  is_active: boolean;
}

interface SchemeFormProps {
  scheme?: Scheme | null;
  onSuccess: () => void;
  onCancel: () => void;
}

export const SchemeForm = ({ scheme, onSuccess, onCancel }: SchemeFormProps) => {
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    name: scheme?.name || "",
    description: scheme?.description || "",
    eligibility: scheme?.eligibility || "",
    benefits: scheme?.benefits || "",
    how_to_apply: scheme?.how_to_apply || "",
    website: scheme?.website || "",
    category: scheme?.category || "",
    is_active: scheme?.is_active ?? true,
  });
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    const payload = {
      name: formData.name,
      description: formData.description || null,
      eligibility: formData.eligibility || null,
      benefits: formData.benefits || null,
      how_to_apply: formData.how_to_apply || null,
      website: formData.website || null,
      category: formData.category || null,
      is_active: formData.is_active,
    };

    let error;
    if (scheme?.id) {
      const result = await supabase
        .from('schemes')
        .update(payload)
        .eq('id', scheme.id);
      error = result.error;
    } else {
      const result = await supabase
        .from('schemes')
        .insert(payload);
      error = result.error;
    }

    setLoading(false);

    if (error) {
      toast({
        title: "Error",
        description: error.message,
        variant: "destructive",
      });
    } else {
      toast({
        title: scheme ? "Updated" : "Created",
        description: `Scheme has been ${scheme ? "updated" : "added"} successfully`,
      });
      onSuccess();
    }
  };

  return (
    <div className="glass-card rounded-xl p-6">
      <div className="flex items-center gap-3 mb-6">
        <Button variant="ghost" size="icon" onClick={onCancel}>
          <ArrowLeft className="w-4 h-4" />
        </Button>
        <h2 className="font-heading text-xl font-semibold text-foreground">
          {scheme ? "Edit Scheme" : "Add New Scheme"}
        </h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Name *
            </label>
            <Input
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Category
            </label>
            <Input
              value={formData.category}
              onChange={(e) => setFormData({ ...formData, category: e.target.value })}
              placeholder="e.g., Health, Financial Aid"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-foreground mb-2">
            Description
          </label>
          <Textarea
            value={formData.description}
            onChange={(e) => setFormData({ ...formData, description: e.target.value })}
            rows={3}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-foreground mb-2">
            Eligibility Criteria
          </label>
          <Textarea
            value={formData.eligibility}
            onChange={(e) => setFormData({ ...formData, eligibility: e.target.value })}
            rows={2}
            placeholder="Who is eligible for this scheme?"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-foreground mb-2">
            Benefits
          </label>
          <Textarea
            value={formData.benefits}
            onChange={(e) => setFormData({ ...formData, benefits: e.target.value })}
            rows={2}
            placeholder="What benefits does this scheme provide?"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-foreground mb-2">
            How to Apply
          </label>
          <Textarea
            value={formData.how_to_apply}
            onChange={(e) => setFormData({ ...formData, how_to_apply: e.target.value })}
            rows={2}
            placeholder="Steps to apply for this scheme"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-foreground mb-2">
            Website
          </label>
          <Input
            value={formData.website}
            onChange={(e) => setFormData({ ...formData, website: e.target.value })}
            placeholder="https://..."
          />
        </div>

        <div className="flex items-center gap-3">
          <Switch
            checked={formData.is_active}
            onCheckedChange={(checked) => setFormData({ ...formData, is_active: checked })}
          />
          <label className="text-sm text-foreground">Active</label>
        </div>

        <div className="flex gap-3 pt-4">
          <Button type="button" variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button type="submit" disabled={loading}>
            {loading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            {scheme ? "Update Scheme" : "Add Scheme"}
          </Button>
        </div>
      </form>
    </div>
  );
};
