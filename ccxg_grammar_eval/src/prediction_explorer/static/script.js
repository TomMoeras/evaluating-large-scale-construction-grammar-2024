document.getElementById("micro-eval-info-btn").addEventListener("click", function () {
    var microEvalInfo = document.getElementById("micro-eval-info-container");
    if (microEvalInfo.style.display === "block") {
        microEvalInfo.style.display = "none";
    } else {
        microEvalInfo.style.display = "block";
    }
});

// Add event listener for the performance parameters button
document.getElementById("performance-parameters-btn").addEventListener("click", function () {
    var performanceParametersContainer = document.getElementById("performance-parameters-container");
    if (performanceParametersContainer.style.display === "block") {
        performanceParametersContainer.style.display = "none";
    } else {
        performanceParametersContainer.style.display = "block";
    }
});

document.addEventListener('DOMContentLoaded', function () {
    const grammarSelect = document.getElementById('grammar-select');
    const predictionContainer = document.getElementById('prediction-container');

    const sentenceInput = document.getElementById('sentence-input');
    const sentencesDatalist = document.getElementById('sentences');

    fetch('/get_sentences')
        .then(response => response.json())
        .then(sentences => {
            for (const sentenceId in sentences) {
                const option = document.createElement('option');
                option.value = `${sentenceId}: ${sentences[sentenceId]}`;
                sentencesDatalist.appendChild(option);
            }
        });

    sentenceInput.addEventListener('input', function () {
        const selectedOption = [...sentencesDatalist.options].find(option => option.value === this.value);
        const selectedSentenceId = selectedOption ? selectedOption.value.split(':')[0] : null;
        const performanceParametersBtn = document.getElementById("performance-parameters-btn");
        performanceParametersBtn.disabled = false;
        // Add event listener for the performance parameters button
        performanceParametersBtn.addEventListener('click', function () {
            const selectedSentenceId = sentenceInput.value.split(':')[0];

            if (selectedSentenceId) {
                fetch(`/performance_parameters/${selectedSentenceId}`)
                    .then(response => response.text())
                    .then(htmlTable => {
                        const performanceParametersContainer = document.getElementById("performance-parameters-container");
                        performanceParametersContainer.innerHTML = htmlTable;
                    });
            }
        });
        if (selectedSentenceId) {
            grammarSelect.disabled = false;
            fetch(`/grammar_info/${selectedSentenceId}`)
                .then(response => response.json())
                .then(grammars => {
                    while (grammarSelect.options.length > 1) {
                        grammarSelect.remove(1);
                    }

                    for (const grammar of grammars) {
                        const option = document.createElement('option');
                        option.value = grammar.grammar_id;
                        option.text = `Grammar ID: ${grammar.grammar_id}, F1: ${grammar.micro_evaluation_info.f1_score}, Config: ${JSON.stringify(grammar.config)}`;
                        grammarSelect.add(option);
                    }
                });
            showPlot(selectedSentenceId);
        } else {
            grammarSelect.disabled = true;
            performanceParametersBtn.disabled = true;
        }
    });

    const plotOptions = document.getElementsByName("plot-option");
    for (const option of plotOptions) {
        option.addEventListener("change", function () {
            const selectedSentenceId = sentenceInput.value.split(':')[0];
            if (selectedSentenceId) {
                showPlot(selectedSentenceId);
            }
        });
    }

    function showPlot(sentence_id) {
        const f1PlotOption = document.getElementById("f1-plot-option");
        const timePlotOption = document.getElementById("time-plot-option");

        let plotApi;
        if (f1PlotOption.checked) {
            plotApi = `/f1_score_plot/${sentence_id}`;
        } else if (timePlotOption.checked) {
            plotApi = `/time_sentence_plot/${sentence_id}`;
        }

        fetch(plotApi)
            .then(response => response.json())
            .then(graphJSON => {
                let plotElement = document.getElementById("plot");
                Plotly.react(plotElement, graphJSON.data, graphJSON.layout);
            });
    }

    grammarSelect.addEventListener('change', function () {
        const selectedGrammarId = this.value;
        const selectedSentenceId = sentenceInput.value.split(':')[0];

        if (selectedGrammarId && selectedSentenceId) {
            microEvalInfoBtn.disabled = false;
            fetch(`/visualize_prediction/${selectedGrammarId}/${selectedSentenceId}`)
                .then(response => response.json())
                .then(data => {
                    const tableHtml = data.table_html;
                    const displacyHtml = data.displacy_html;
                    predictionContainer.innerHTML = `
                        <div class="container">
                            <div class="column">
                                <h2>Table:</h2>
                                <div class="tables-container">${tableHtml}</div>
                            </div>
                            <div class="column">
                                <h2>Displacy:</h2>
                                <div class="displacy-container">${displacyHtml}</div>
                            </div>
                        </div>`;
                });
        } else {
            microEvalInfoBtn.disabled = true;
        }
    });

    const microEvalInfoBtn = document.getElementById('micro-eval-info-btn');
    const microEvalInfoContainer = document.getElementById('micro-eval-info-container');

    microEvalInfoBtn.addEventListener('click', function () {
        const selectedGrammarId = grammarSelect.value;
        const selectedSentenceId = sentenceInput.value.split(':')[0];

        if (selectedGrammarId && selectedSentenceId) {
            fetch(`/get_micro_evaluation_info/${selectedGrammarId}/${selectedSentenceId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.message) {
                        microEvalInfoContainer.innerHTML = `<pre>${data.message}</pre>`;
                    } else {
                        microEvalInfoContainer.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
                    }
                });
        }
    });

});