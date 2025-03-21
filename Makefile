DATA_DIR = data
FIGURES_DIR = results/figures
ANIMATIONS_DIR = results/animations

ENTRYPOINT ?= uv run

QUALITY ?= low

ifeq ($(QUALITY),low)
	QUALITY_PARAMS_COMPARE_EIGENSOLVER_RUNTIME = --repeats 5 --timeout "0.25" --quality-label $(QUALITY)
	QUALITY_PARAMS_COMPARE_EIGENSOLVER_RESULTS = --max-n 66 --quality-label $(QUALITY)
	QUALITY_PARAMS_EIGENSPECTRUM_BY_LENGTH = --n-at-unit-length 50 --quality-label $(QUALITY)
	QUALITY_PARAMS_EIGENSPECTRUMS_BOTH = --min-n 20 --max-n 50 --quality-label $(QUALITY)
	QUALITY_PARAMS_WAVE_ANIMATION = --domain circle --n 50 --animation-speed 1 --repeats 3 --fps 10 --dpi 100 --quality-label $(QUALITY)
else ifeq ($(QUALITY),high)
	QUALITY_PARAMS_COMPARE_EIGENSOLVER_RUNTIME = --repeats 30 --timeout "4.0" --quality-label $(QUALITY)
	QUALITY_PARAMS_COMPARE_EIGENSOLVER_RESULTS = --max-n 91 --quality-label $(QUALITY)
	QUALITY_PARAMS_EIGENSPECTRUM_BY_LENGTH = --n-at-unit-length 100 --quality-label $(QUALITY)
	QUALITY_PARAMS_EIGENSPECTRUMS_BOTH = --min-n 20 --max-n 50 --quality-label $(QUALITY)
	QUALITY_PARAMS_WAVE_ANIMATION = --domain circle --n 500 --animation-speed 1 --repeats 5 --fps 60 --dpi 200 --quality-label $(QUALITY)
else
	$(error Invalid quality specifier: $(QUALITY). Choose 'low' or 'high'.)
endif

FIGURE_NAMES = \
		eigenmodes.pdf \
		compare_results_eigensolvers_$(QUALITY)_quality.pdf \
		eigenfrequency_spectrums_$(QUALITY)_quality.pdf \
		spring_1d_phaseplot.pdf \
		spring_1d_energy.pdf

SERIAL_FIGURE_NAMES = \
		compare_runtime_eigensolvers_$(QUALITY)_quality.pdf

FIGURES = $(patsubst %, $(FIGURES_DIR)/%, $(FIGURE_NAMES))
SERIAL_FIGURES = $(patsubst %, $(FIGURES_DIR)/%, $(SERIAL_FIGURE_NAMES))

CIRCULAR_DRUM_ANIMATION_KS ?= 0 1 3 5 6 8 38 50 51
ANIMATION_NAMES = \
		$(patsubst %, circular_drum_k_%_$(QUALITY)_quality.mp4, $(CIRCULAR_DRUM_ANIMATION_KS))
ANIMATIONS = $(patsubst %, $(ANIMATIONS_DIR)/%, $(ANIMATION_NAMES))


.PHONY: all serial clean


all: $(FIGURES) $(ANIMATIONS) 

serial: $(SERIAL_FIGURES)

$(FIGURES_DIR)/eigenmodes.pdf: \
			src/scicomp/cli/plots/eigenmodes.py \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) scicomp plot eigenmodes

$(FIGURES_DIR)/compare_runtime_eigensolvers_$(QUALITY)_quality.pdf: \
			src/scicomp/cli/plots/measure_sparse_dense_eig_runtime.py \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) scicomp plot compare-eigensolver-runtime $(QUALITY_PARAMS_COMPARE_EIGENSOLVER_RUNTIME)

$(FIGURES_DIR)/compare_results_eigensolvers_$(QUALITY)_quality.pdf: \
			src/scicomp/cli/plots/measure_relative_error_sparse_eigenvalues.py \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) scicomp plot compare-eigensolver-results $(QUALITY_PARAMS_COMPARE_EIGENSOLVER_RESULTS)

$(FIGURES_DIR)/eigenfrequency_spectrums_$(QUALITY)_quality.pdf: \
			src/scicomp/cli/plots/eigenfrequency_spectrums_both.py \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) scicomp plot eigenspectrums $(QUALITY_PARAMS_EIGENSPECTRUMS_BOTH)

$(FIGURES_DIR)/spring_1d_phaseplot.pdf: \
			src/scicomp/cli/plots/spring_phaseplot.py \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) scicomp plot spring-phaseplot

$(FIGURES_DIR)/spring_1d_energy.pdf: \
			src/scicomp/cli/plots/spring_energy.py \
			| $(FIGURES_DIR)
	$(ENTRYPOINT) scicomp plot spring-energy

$(ANIMATIONS_DIR)/circular_drum_k_%_$(QUALITY)_quality.mp4: \
			src/scicomp/cli/animations/create_wave_animation.py \
			| $(ANIMATIONS_DIR)
	$(ENTRYPOINT) scicomp animate eigenmode --k $* $(QUALITY_PARAMS_WAVE_ANIMATION)



$(FIGURES_DIR):
	mkdir -p $@

$(ANIMATIONS_DIR):
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf results
