let interval = 0;
let minutes = 0;
let seconds = 0;
let centiseconds = 0;
let timerRef = document.querySelector('.timer');
let editTime = document.querySelector('#edit_time');


function start() {
    if (interval != null) {
        clearInterval(interval)
    }
    interval = setInterval(displayTimer, 10);
}

function stop() {
    clearInterval(interval)
    editTime.value = minutes * 60 + seconds + centiseconds / 100;
}

function displayTimer() {
    centiseconds += 1;
    if (centiseconds == 100) {
        centiseconds %= 100;
        seconds += 1;
    }
    if (seconds == 60) {
        seconds %= 60;
        minutes += 1;
    }
    timerRef.innerHTML = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(centiseconds).padStart(2, '0')}`;
}

function demo_1() {
    setTimeout(function() {
        accuracy = document.querySelector('input[name="clinical"]:checked').value;
        impressionText = document.getElementById('edited_impression');
        if (accuracy == 'frankly misleading') {
            impressionText.value = '1. Small bilateral adrenal nodules measuring up to 13 mm on the right and 10 mm on the left suspicious for metastatic disease.'
        }
        if (accuracy == 'inaccurate') {
            impressionText.value = '1. Evidence of metastatic disease at the chest.';
        }
        if (accuracy == 'mostly accurate') {
            impressionText.value = '1. No evidence of disease to the chest.';
        }
        if (accuracy == 'accurate') {
            impressionText.value = '1. No evidence of metastatic disease to the chest.\n2. Overall decreased extent of waxing and waning clustered centrilobular nodules and tree-in-bud opacities, compatible with ongoing but improving chronic infection.'
        }
    }, 100);
}


function demo_2() {
    setTimeout(function() {
        accuracy = document.querySelector('input[name="grammatical"]:checked').value;
        impressionText = document.getElementById('edited_impression');
        if (accuracy == 'frankly misleading') {
            impressionText.value = '<pad> no evidence of metastatic disease to the chest decreased extent of centrilobular and tree-in-bud, compatible with ongoing but improving chronic infection.. </s>'
        }
        if (accuracy == 'inaccurate') {
            impressionText.value = '1. no evidence disease at the chest. overall decreased extent of waning centrilobular and tree-in-bud opacities, compatible with ongoing but infection..';
        }
        if (accuracy == 'mostly accurate') {
            impressionText.value = '1. No evidence disease the chest.\n2. Overall decreased extent of waning centrilobular nodules and tree-in-bud opacities, ongoing but improving infection.';
        }
        if (accuracy == 'accurate') {
            impressionText.value = '1. No evidence of metastatic disease to the chest.\n2. Overall decreased extent of waxing and waning clustered centrilobular nodules and tree-in-bud opacities, compatible with ongoing but improving chronic infection.'
        }
    }, 100);
}

function demo_3() {
    setTimeout(function() {
        accuracy = document.querySelector('input[name="quality"]:checked').value;
        impressionText = document.getElementById('edited_impression');
        if (accuracy == 'very unsatisfactory') {
            impressionText.value = '1. Overall decreased extent of waxing and waning clustered centrilobular nodules and tree-in-bud opacities, for example increased within the lingula but decreased within the right upper lobe. Unchanged more discrete juxtapleural solid nodules, including a 4 mm right lower lobe nodule (series 6, image 194), unchanged dating back to 3/6/2018, and favored to be benign.\n2. Unchanged heart size. No pericardial effusion. Mild coronary calcifications. Normal caliber ascending aorta and main pulmonary artery. Interval removal of right internal jugular chest port, with residual fibrin sheath along the catheter course within the right internal jugular and brachiocephalic veins.\n3. No evidence of metastatic disease to the chest.'
        }
        if (accuracy == 'unsatisfactory') {
            impressionText.value = '1. Overall decreased extent of waxing and waning clustered centrilobular nodules and tree-in-bud opacities, for example increased within the lingula but decreased within the right upper lobe. Unchanged more discrete juxtapleural solid nodules, including a 4 mm right lower lobe nodule (series 6, image 194), unchanged dating back to 3/6/2018, and favored to be benign.\n2. No evidence of metastatic disease to the chest.'
        }
        if (accuracy == 'mostly satisfactory') {
            impressionText.value = '1. No evidence of metastatic disease to the chest.\n2. Overall decreased extent of waxing and waning clustered centrilobular nodules and tree-in-bud opacities, for example increased within the lingula but decreased within the right upper lobe.';
        }
        if (accuracy == 'satisfactory') {
            impressionText.value = '1. No evidence of metastatic disease to the chest.\n2. Overall decreased extent of waxing and waning clustered centrilobular nodules and tree-in-bud opacities, compatible with ongoing but improving chronic infection.'
        }
    }, 100);
}