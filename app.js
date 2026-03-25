let rnnData = null;
let networkNodes = [];
let networkLinks = [];
let h_state = [];
let isPlaying = false;
let loopEvent = null;

let allRnnData = null;
let currentNetworkIndex = 0;
const networkSelect = document.getElementById('network-select');

// UI Elements
const startBtn = document.getElementById('start-btn');
const statusEl = document.getElementById('status');
const bpmSlider = document.getElementById('bpm-slider');
const bpmVal = document.getElementById('bpm-val');
const noiseSlider = document.getElementById('noise-slider');
const noiseVal = document.getElementById('noise-val');
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdVal = document.getElementById('threshold-val');
const autoSwitchCheckbox = document.getElementById('auto-switch');
const autoNoiseCheckbox = document.getElementById('auto-noise');
const autoThresholdCheckbox = document.getElementById('auto-threshold');
const iosWarning = document.getElementById('ios-warning');

// Detect iOS devices to display the hardware mute warning
const isIOS = [
    'iPad Simulator',
    'iPhone Simulator',
    'iPod Simulator',
    'iPad',
    'iPhone',
    'iPod'
].includes(navigator.platform)
    || (navigator.userAgent.includes("Mac") && "ontouchend" in document);

if (isIOS && iosWarning) {
    iosWarning.style.display = 'block';
}

let autoSwitchInterval = null;
let autoNoiseInterval = null;
let autoThresholdInterval = null;

function startAutoThreshold() {
    clearInterval(autoThresholdInterval);
    const sampleAndSet = () => {
        if (!isPlaying || !autoThresholdCheckbox.checked) return;
        noteThreshold = 0.2 + Math.random() * 0.3; // 0.2..0.5
        thresholdSlider.value = noteThreshold.toFixed(2);
        thresholdVal.textContent = noteThreshold.toFixed(2);
    };
    // Run it instantly on engagement so there is no 15s visual delay
    sampleAndSet();
    autoThresholdInterval = setInterval(sampleAndSet, 15000);
}

autoThresholdCheckbox.addEventListener('change', (e) => {
    if (e.target.checked && isPlaying) {
        startAutoThreshold();
    } else {
        clearInterval(autoThresholdInterval);
    }
});

function startAutoNoise() {
    clearInterval(autoNoiseInterval);
    autoNoiseInterval = setInterval(() => {
        if (!isPlaying || !autoNoiseCheckbox.checked) return;
        noiseLevel += 0.1;
        if (noiseLevel > 0.5001) {
            noiseLevel = 0.0;
        }
        noiseSlider.value = noiseLevel;
        noiseVal.textContent = noiseLevel.toFixed(2);
    }, 10000);
}

autoNoiseCheckbox.addEventListener('change', (e) => {
    if (e.target.checked && isPlaying) {
        startAutoNoise();
    } else {
        clearInterval(autoNoiseInterval);
    }
});

function startAutoSwitch() {
    clearInterval(autoSwitchInterval);
    autoSwitchInterval = setInterval(() => {
        if (!isPlaying || !autoSwitchCheckbox.checked) return;
        currentNetworkIndex = (currentNetworkIndex + 1) % allRnnData.networks.length;
        networkSelect.value = currentNetworkIndex;
        loadSelectedNetwork();
        statusEl.textContent = `Playing Rank ${currentNetworkIndex + 1} Variation... (Auto-Switched)`;
    }, 60000);
}

autoSwitchCheckbox.addEventListener('change', (e) => {
    if (e.target.checked && isPlaying) {
        startAutoSwitch();
    } else {
        clearInterval(autoSwitchInterval);
    }
});

let currentBPM = parseInt(bpmSlider.value);
let noiseLevel = parseFloat(noiseSlider.value);
let noteThreshold = parseFloat(thresholdSlider.value);

let activeNotes = new Set();

bpmSlider.addEventListener('input', (e) => {
    currentBPM = parseInt(e.target.value);
    bpmVal.textContent = currentBPM;
    if (isPlaying) {
        Tone.Transport.bpm.value = currentBPM;
    }
});

noiseSlider.addEventListener('input', (e) => {
    noiseLevel = parseFloat(e.target.value);
    noiseVal.textContent = noiseLevel;
});

thresholdSlider.addEventListener('input', (e) => {
    noteThreshold = parseFloat(e.target.value);
    thresholdVal.textContent = noteThreshold;
});

// Setup Tone.js Synthesis
const limiter = new Tone.Limiter(-10).toDestination();
const compressor = new Tone.Compressor({
    threshold: -30,
    ratio: 4,
    attack: 0.03,
    release: 0.1
}).connect(limiter);

// Filter for warmer sound
const filterLow = new Tone.Filter(1800, "lowpass").connect(compressor);

const synth = new Tone.PolySynth(Tone.Synth, {
    oscillator: {
        type: "sine"
    },
    envelope: {
        attack: 0.05,
        decay: 0.3,
        sustain: 0.4,
        release: 1.2
    },
    volume: -12
}).connect(filterLow);

// Add some mild reverb
const reverb = new Tone.Reverb(2.0).connect(filterLow);
synth.connect(reverb);


// Setup D3 Force Graph
const svg = d3.select("#network-graph");
let width = window.innerWidth;
let height = window.innerHeight;

// Add Glow Filter
const defs = svg.append("defs");
const filter = defs.append("filter").attr("id", "glow");
filter.append("feGaussianBlur").attr("stdDeviation", "3.5").attr("result", "coloredBlur");
const feMerge = filter.append("feMerge");
feMerge.append("feMergeNode").attr("in", "coloredBlur");
feMerge.append("feMergeNode").attr("in", "SourceGraphic");

const gLinks = svg.append("g").attr("class", "links");
const gNodes = svg.append("g").attr("class", "nodes");

let simulation;

// Load Data
fetch('rnn_top20.json')
    .then(res => res.json())
    .then(data => {
        allRnnData = data;

        // Populate select
        networkSelect.innerHTML = '';
        data.networks.forEach((net, idx) => {
            const opt = document.createElement('option');
            opt.value = idx;
            opt.textContent = `Rank ${idx + 1} (Loss: ${net.loss.toFixed(4)})`;
            networkSelect.appendChild(opt);
        });

        networkSelect.addEventListener('change', (e) => {
            currentNetworkIndex = parseInt(e.target.value);
            loadSelectedNetwork();
        });

        loadSelectedNetwork();
    })
    .catch(err => {
        statusEl.textContent = 'Error loading RNN: ' + err.message;
    });

function loadSelectedNetwork() {
    const wasPlaying = isPlaying;
    if (wasPlaying) {
        Tone.Transport.pause();
    }

    const net = allRnnData.networks[currentNetworkIndex];
    rnnData = {
        W: net.W,
        b: net.b,
        h0: net.h0,
        O_notes: allRnnData.O_notes,
        T_steps: allRnnData.T_steps,
        target_notes: allRnnData.target_notes,
        notes_map: allRnnData.notes_map
    };
    statusEl.textContent = 'RNN Ready. ' + rnnData.W.length + ' neurons.';

    // reset graph
    networkNodes = [];
    networkLinks = [];
    if (simulation) {
        simulation.stop();
        gNodes.selectAll("*").remove();
        gLinks.selectAll("*").remove();
    }

    synth.releaseAll();
    activeNotes.clear();
    initializeNetwork();

    if (wasPlaying) {
        setTimeout(() => {
            if (isPlaying) Tone.Transport.start();
        }, 400);
    }
}

function initializeNetwork() {
    const N = rnnData.W.length;
    h_state = [...rnnData.h0]; // starting state

    // Create nodes
    // First O_notes are outputs
    const reverse_map = {};
    for (let note in rnnData.notes_map) {
        reverse_map[rnnData.notes_map[note]] = note;
    }

    window.nodeMap = {};
    for (let i = 0; i < N; i++) {
        let isOutput = i < rnnData.O_notes;

        let isPruned = true;
        if (rnnData.b[i] !== 0) isPruned = false;
        if (isPruned) {
            for (let j = 0; j < N; j++) {
                // If it has incoming or outgoing connections, it's alive!
                if (rnnData.W[i][j] !== 0 || rnnData.W[j][i] !== 0) {
                    isPruned = false;
                    break;
                }
            }
        }

        if (isPruned && !isOutput) {
            window.nodeMap[i] = null;
            continue; // Skip adding to D3 simulation layer structurally completely
        }

        let nObj = {
            id: i,
            isOutput: isOutput,
            note: isOutput ? reverse_map[i] : null,
            activation: h_state[i]
        };
        networkNodes.push(nObj);
        window.nodeMap[i] = nObj;
    }

    // Create links (only for connections > 0.01 to keep rendering fast, or all non-zero)
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            let weight = rnnData.W[i][j];
            if (Math.abs(weight) > 0.01) {
                networkLinks.push({
                    source: i,
                    target: j,
                    weight: weight
                });
            }
        }
    }

    // D3 setup
    simulation = d3.forceSimulation(networkNodes)
        .force("link", d3.forceLink(networkLinks).id(d => d.id).distance(150))
        .force("charge", d3.forceManyBody().strength(-200))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("x", d3.forceX(width / 2).strength(0.05))
        .force("y", d3.forceY(height / 2).strength(0.05));

    const link = gLinks.selectAll("line")
        .data(networkLinks)
        .join("line")
        .attr("class", "link")
        .attr("stroke", d => d.weight > 0 ? "#4a90e2" : "#e05252")
        .attr("stroke-width", d => Math.min(Math.abs(d.weight) * 3, 5))
        .attr("stroke-opacity", 0.4);

    const drag = simulation => {
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        return d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended);
    };

    const node = gNodes.selectAll("g")
        .data(networkNodes)
        .join("g")
        .attr("class", "node")
        .call(drag(simulation));

    node.append("circle")
        .attr("r", d => d.isOutput ? 14 : 6)
        .attr("fill", "#333")
        .style("filter", "url(#glow)");

    node.append("text")
        .attr("dx", 16)
        .attr("dy", 4)
        .text(d => d.isOutput ? d.note : "");

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node.attr("transform", d => `translate(${d.x},${d.y})`);
    });
}

async function togglePlayback(e) {
    if (e) e.preventDefault();
    await Tone.start();
    if (!isPlaying) {
        Tone.Transport.bpm.value = currentBPM;
        Tone.Transport.start();
        loopEvent = Tone.Transport.scheduleRepeat((time) => {
            stepRNN(time);
        }, "8n");
        isPlaying = true;
        startBtn.textContent = 'Stop Playing';
        statusEl.textContent = `Playing Rank ${currentNetworkIndex + 1} Variation...`;

        if (autoSwitchCheckbox.checked) startAutoSwitch();
        if (autoNoiseCheckbox.checked) startAutoNoise();
        if (autoThresholdCheckbox.checked) startAutoThreshold();
    } else {
        Tone.Transport.stop();
        Tone.Transport.clear(loopEvent);
        synth.releaseAll();
        activeNotes.clear();
        isPlaying = false;
        clearInterval(autoSwitchInterval);
        clearInterval(autoNoiseInterval);
        clearInterval(autoThresholdInterval);
        startBtn.textContent = 'Start Playing';
    }
}

startBtn.addEventListener('click', togglePlayback);
startBtn.addEventListener('touchstart', togglePlayback);

window.addEventListener('resize', () => {
    width = window.innerWidth;
    height = window.innerHeight;
    if (simulation) {
        simulation.force("center", d3.forceCenter(width / 2, height / 2));
        simulation.alpha(0.3).restart();
    }
});

// Ensure color clamping
function getActivationColor(act, isOutput) {
    // act usually [-1, 1], shifted to [0,1]
    let factor = (act + 1) / 2;
    factor = Math.max(0, Math.min(1, factor));
    if (isOutput) {
        // gradient for output: dark to bright yellow/orange
        const r = Math.floor(50 + 205 * factor);
        const g = Math.floor(50 + 150 * factor);
        return `rgb(${r},${g},50)`;
    } else {
        // gradient for internal: dark to bright cyan
        const b = Math.floor(50 + 205 * Math.abs(act)); // absolute activity makes it glow
        return `rgb(50,150,${b})`;
    }
}

function getActivationRadius(act, isOutput) {
    let factor = (act + 1) / 2;
    if (isOutput) return 10 + 10 * factor;
    return 6 + 6 * Math.abs(act);
}

function stepRNN(time) {
    if (!rnnData) return;
    const N = rnnData.W.length;
    const next_h = new Array(N).fill(0);

    // Matrix multiplication: h_next[j] = b[j] + sum(h[i] * W[i][j])
    for (let i = 0; i < N; i++) {
        let hi = h_state[i];
        if (hi === 0) continue; // skip 0
        for (let j = 0; j < N; j++) {
            let weight = rnnData.W[i][j];
            if (weight !== 0) {
                next_h[j] += hi * weight;
            }
        }
    }

    // 1. Add static bias
    for (let j = 0; j < N; j++) {
        next_h[j] += rnnData.b[j];
    }

    // 2. Compute dynamic LayerNorm statistics over all neurons
    let sumAct = 0;
    for (let j = 0; j < N; j++) {
        sumAct += next_h[j];
    }
    const meanAct = sumAct / N;

    let sqSumAct = 0;
    for (let j = 0; j < N; j++) {
        sqSumAct += Math.pow(next_h[j] - meanAct, 2);
    }
    const stdDev = Math.sqrt((sqSumAct / N) + 1e-5);

    // 3. Normalize all neurons
    for (let j = 0; j < N; j++) {
        next_h[j] = (next_h[j] - meanAct) / stdDev;
    }

    // 4. Apply tanh activation and add stochasticity
    let currentActive = new Set();
    for (let j = 0; j < N; j++) {
        let baseAct = next_h[j];

        // 1. Core State: Pure deterministic recurrent state tracking identically PyTorch trained constraints
        let cleanAct = Math.tanh(baseAct);
        h_state[j] = cleanAct;

        // 2. Visual & Output State: Add stochasticity purely dynamically without permanently damaging the recurrence
        let noisyAct = baseAct;
        if (noiseLevel > 0) {
            noisyAct += (Math.random() * 2 - 1) * noiseLevel;
        }
        let finalOutputAct = Math.tanh(noisyAct);

        let nodeRef = window.nodeMap[j];
        if (nodeRef) {
            nodeRef.activation = finalOutputAct;
        }

        // Check output notes explicitly on the noisy outputs locally
        if (j < rnnData.O_notes) {
            let prob = (finalOutputAct + 1) / 2;
            if (prob > noteThreshold && nodeRef) {
                currentActive.add(nodeRef.note);
            }
        }
    }

    let toAttack = [];
    currentActive.forEach(note => {
        if (!activeNotes.has(note)) toAttack.push(note);
    });

    let toRelease = [];
    activeNotes.forEach(note => {
        if (!currentActive.has(note)) toRelease.push(note);
    });

    if (toAttack.length > 0) synth.triggerAttack(toAttack, time);
    if (toRelease.length > 0) synth.triggerRelease(toRelease, time);

    activeNotes = currentActive;

    // Animate visualizer via D3 (must be run on main thread sync to drawing)
    // We defer to requestAnimationFrame to decouple heavy DOM ops from audio thread if needed,
    // though for 120 nodes it's okay.
    requestAnimationFrame(() => {
        gNodes.selectAll("circle")
            .attr("fill", d => getActivationColor(d.activation, d.isOutput))
            .attr("r", d => getActivationRadius(d.activation, d.isOutput));

        gLinks.selectAll("line")
            .attr("stroke-opacity", d => {
                let act = (h_state[d.source] + 1) / 2;
                return 0.1 + 0.9 * act; // pulse links based on source activation
            });
    });
}
