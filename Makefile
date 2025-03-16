DATA_DIR = data
FIGURES_DIR = results/figures
ANIMATIONS_DIR = results/animations

ENTRYPOINT ?= uv run

QUALITY ?= low

ifeq ($(QUALITY),low)
	QUALITY_PARAMS_COMPARE_EIGENSOLVER_RUNTIME = --repeats 5 --timeout "0.25" --quality-label $(QUALITY)
	QUALITY_PARAMS_COMPARE_EIGENSOLVER_RESULTS = --max-n 65 --quality-label $(QUALITY)
	QUALITY_PARAMS_EIGENSPECTRUM_BY_LENGTH = --n 50 --quality-label $(QUALITY)
	QUALITY_PARAMS_EIGENSPECTRUM_BY_N = --max-n 100 --quality-label $(QUALITY)
	QUALITY_PARAMS_WAVE_ANIMATION = --n 50 --animation-speed 10 --repeats 3 --fps 10 --dpi 100 --quality-label $(QUALITY)
else ifeq ($(QUALITY),high)
	QUALITY_PARAMS_COMPARE_EIGENSOLVER_RUNTIME = --repeats 30 --timeout "4.0" --quality-label $(QUALITY)
	QUALITY_PARAMS_COMPARE_EIGENSOLVER_RESULTS = --max-n 81 --quality-label $(QUALITY)
	QUALITY_PARAMS_EIGENSPECTRUM_BY_LENGTH = --n 200 --quality-label $(QUALITY)
	QUALITY_PARAMS_EIGENSPECTRUM_BY_N = --max-n 150 --quality-label $(QUALITY)
	QUALITY_PARAMS_WAVE_ANIMATION = --n 500 --animation-speed 10 --repeats 5 --fps 60 --dpi 200 --quality-label $(QUALITY)
else
	$(error Invalid quality specifier: $(QUALITY). Choose 'low' or 'high'.)
endif

FIGURE_NAMES = \
		eigenmodes.pdf \
		compare_runtime_eigensolvers_$(QUALITY)_quality.pdf \
		compare_results_eigensolvers_$(QUALITY)_quality.pdf \
		eigenfrequency_spectrum_by_length_$(QUALITY)_quality.pdf \
		eigenfrequency_spectrum_by_n_$(QUALITY)_quality.pdf
FIGURES = $(patsubst %, $(FIGURES_DIR)/%, $(FIGURE_NAMES))

CIRCULAR_DRUM_ANIMATION_KS ?= 0 1 3 5 6 8 38 50 51
ANIMATION_NAMES = \
		$(patsubst %, circular_drum_k_%_$(QUALITY)_quality.mp4, $(CIRCULAR_DRUM_ANIMATION_KS))
ANIMATIONS = $(patsubst %, $(ANIMATIONS_DIR)/%, $(ANIMATION_NAMES))


all: $(FIGURES) $(ANIMATIONS)

$(FIGURES_DIR)/eigenmodes.pdf: \
			src/scicomp/eig_val_calc/circle.py \
			scripts/plot_eigenmodes.py \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) scripts/plot_eigenmodes.py

$(FIGURES_DIR)/compare_runtime_eigensolvers_$(QUALITY)_quality.pdf: \
			src/scicomp/eig_val_calc/circle.py \
			scripts/measure_sparse_dense_eig_runtime.py \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) scripts/measure_sparse_dense_eig_runtime.py $(QUALITY_PARAMS_COMPARE_EIGENSOLVER_RUNTIME)

$(FIGURES_DIR)/compare_results_eigensolvers_$(QUALITY)_quality.pdf: \
			src/scicomp/eig_val_calc/circle.py \
			scripts/measure_relative_error_sparse_eigenvalues.py \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) scripts/measure_relative_error_sparse_eigenvalues.py $(QUALITY_PARAMS_COMPARE_EIGENSOLVER_RESULTS)

$(FIGURES_DIR)/eigenfrequency_spectrum_by_length_$(QUALITY)_quality.pdf: \
			src/scicomp/eig_val_calc/circle.py \
			scripts/plot_eigenfrequency_spectrums.py \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) scripts/plot_eigenfrequency_spectrums.py $(QUALITY_PARAMS_EIGENSPECTRUM_BY_LENGTH)

$(FIGURES_DIR)/eigenfrequency_spectrum_by_n_$(QUALITY)_quality.pdf: \
			src/scicomp/eig_val_calc/circle.py \
			scripts/plot_eigenfrequency_spectrums_by_n.py \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) scripts/plot_eigenfrequency_spectrums_by_n.py $(QUALITY_PARAMS_EIGENSPECTRUM_BY_N)

$(ANIMATIONS_DIR)/circular_drum_k_%_$(QUALITY)_quality.mp4: \
			src/scicomp/eig_val_calc/circle.py \
			scripts/create_wave_animation.py \
			| $(ANIMATIONS_DIR)
	$(ENTRYPOINT) scripts/create_wave_animation.py --k $* $(QUALITY_PARAMS_WAVE_ANIMATION)



$(FIGURES_DIR):
	mkdir -p $@

$(ANIMATIONS_DIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf results
