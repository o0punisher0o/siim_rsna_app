const { createApp } = Vue;

createApp({
  data() {
    return {
      file: null,
      originalPreview: null,
      result: null,
      loading: false
    };
  },

  computed: {
    explanationText() {
      if (!this.result) return "";

      const c = this.result.predicted_class;

      if (c === "negative")
        return "No strong visual patterns associated with pneumonia were detected.";

      if (c === "typical")
        return "The model detected visual patterns commonly associated with pneumonia.";

      if (c === "atypical")
        return "The model detected unusual patterns that may require further clinical evaluation.";

      return "The model found ambiguous visual patterns and cannot make a confident decision.";
    }
  },

  methods: {
    onFileChange(e) {
      this.file = e.target.files[0];
      this.originalPreview = URL.createObjectURL(this.file);
    },

    async analyze() {
      this.loading = true;
      this.result = null;

      const formData = new FormData();
      formData.append("file", this.file);

      const response = await fetch(
        "http://127.0.0.1:8000/predict?explain=true",
        {
          method: "POST",
          body: formData
        }
      );

      this.result = await response.json();
      this.loading = false;
    }
  }
}).mount("#app");
