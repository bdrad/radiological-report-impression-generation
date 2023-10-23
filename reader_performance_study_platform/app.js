const express = require('express')
const app = express()
const cors = require('cors')
const dotenv = require('dotenv')
const bodyParser = require('body-parser')
const { Sequelize, DataTypes } = require('sequelize')

const { SlimNodeMySQL } = require('slim-node-mysql')

const session = require('express-session');
const flash = require('connect-flash');

dotenv.config()

if(typeof String.prototype.replaceAll === "undefined") {
    String.prototype.replaceAll = function(match, replace) {
       return this.replace(new RegExp(match, 'g'), () => replace);
    }
}

const mySQLString = `mysql://${process.env.DB_USER}:${process.env.DB_PASSWORD}@${process.env.DB_HOST}/${process.env.DB}?reconnect=true`

const database = new Sequelize(mySQLString)

const PORT = 8080

const user = database.define('user', {
    // Model attributes are defined here
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true
    },
    name: {
        type: DataTypes.STRING,
    },
    token: {
        type: DataTypes.STRING,
    },
    current_set: {
        type: DataTypes.STRING,
    },
    
  }, {
    timestamps: false,
  });

const report = database.define('report', {
    // Model attributes are defined here
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true
    },
    accession_number: {
        type: DataTypes.STRING,
    },
    exam: {
        type: DataTypes.STRING,
    },
    clinical_history: {
        type: DataTypes.STRING,
    },
    comparison: {
        type: DataTypes.STRING,
    },
    findings: {
        type: DataTypes.STRING,
    },
    original_impression: {
        type: DataTypes.STRING,
    },
    predicted_impression: {
        type: DataTypes.STRING,
    },
    original: {
        type: DataTypes.TINYINT,
    },
    
  }, {
    timestamps: false,
  });

const evaluation = database.define('evaluation', {
    // Model attributes are defined here
    id: {
      type: DataTypes.INTEGER,
      autoIncrement: true,
      primaryKey: true
    },
    user_id: {
        type: DataTypes.INTEGER,
    },
    report_id: {
        type: DataTypes.INTEGER,
    },
    clinical_accuracy: {
        type: DataTypes.STRING,
    },
    grammatical_accuracy: {
        type: DataTypes.STRING,
    },
    edited_impression: {
        type: DataTypes.STRING,
    },
    stylistic_quality: {
        type: DataTypes.STRING,
    },
    edit_time: {
        type: DataTypes.FLOAT,
    },
    
  }, {
    timestamps: false,
  });


app 
    .set('view engine', 'ejs')

    // Serve static images and CSS files 
    .use(express.static('public'))

    // Parse through the incoming requests
    .use(bodyParser.urlencoded({ extended: false }))
    .use(bodyParser.json())

    .use(function(req, res, next) {
        res.set('Cache-Control', 'no-cache, private, no-store, must-revalidate, max-stale=0, post-check=0, pre-check=0');
        next();
    })

    .use(session({
        secret: 'secret key',
        resave: false,
        saveUninitialized: false
    }))

    .use(flash())

    .get('/', async function(req, res) {
        await database.authenticate()
        res.render('index');
    })

    .get('*', async function (req, res) {
        res.redirect('/')
    })

    .post('/login', async(req, res) => {
        let token = req.body['token']
        const valid_user = await user.findOne({ where: {token: token} })
        if (valid_user && valid_user.current_set == 'demo_1') {
            req.flash('user_id', valid_user.id);
            req.flash('current_set', valid_user.current_set);
            return res.render('demo_1', {
                current_set: 'demo_1',
                user_id: valid_user.id
            });
        }
        if (valid_user && valid_user.current_set == 'demo_2') {
            req.flash('user_id', valid_user.id);
            req.flash('current_set', valid_user.current_set);
            return res.render('demo_2', {
                current_set: 'demo_2',
                user_id: valid_user.id
            });
        }
        if (valid_user && valid_user.current_set == 'demo_3') {
            req.flash('user_id', valid_user.id);
            req.flash('current_set', valid_user.current_set);
            return res.render('demo_3', {
                current_set: 'demo_3',
                user_id: valid_user.id
            });
        }
        if (valid_user && valid_user.current_set == 'demo_4') {
            req.flash('user_id', valid_user.id);
            req.flash('current_set', valid_user.current_set);
            return res.render('demo_4', {
                current_set: 'demo_4',
                user_id: valid_user.id
            });
        }
        if (valid_user) {
            req.flash('user_id', valid_user.id);
            req.flash('current_set', valid_user.current_set);
            res.redirect(307, '/home');
        } else {
            res.redirect('/')
        }
    })

    .post('/demo_1', async(req, res) => {
        user_id = req.body['user_id']
        res.render('demo_2', {
            current_set: 'demo_2',
            user_id: user_id
        })
        await user.update(
            { current_set: 'demo_2' },
            { where: { id: user_id } }
        )
    })

    .post('/demo_2', async(req, res) => {
        user_id = req.body['user_id']
        res.render('demo_3', {
            current_set: 'demo_3',
            user_id: user_id
        })
        await user.update(
            { current_set: 'demo_3' },
            { where: { id: user_id } }
        )
    })

    .post('/demo_3', async(req, res) => {
        user_id = req.body['user_id']
        res.render('demo_4', {
            current_set: 'demo_4',
            user_id: user_id
        })
        await user.update(
            { current_set: 'demo_4' },
            { where: { id: user_id } }
        )
    })

    .post('/demo_4', async(req, res) => {
        user_id = req.body['user_id']
        await user.update(
            { current_set: 1 },
            { where: { id: user_id } }
        )
        req.flash('user_id', user_id);
        req.flash('current_set', 1);
        res.redirect(307, '/home');
    })

    .post('/home', async(req, res) => {
        current_set_elems = req.flash('current_set')
        current_set = current_set_elems[current_set_elems.length - 1]
        user_id = req.flash('user_id')[0]
        current_report = await report.findOne({ where: {id: parseInt(current_set)} }) 
        res.render('home', {
            exam: current_report.exam,
            clinical_history: current_report.clinical_history,
            findings: current_report.findings.replaceAll("\n", '<br/>'),
            comparison: current_report.comparison,
            original_impression: current_report.original_impression,
            predicted_impression: current_report.predicted_impression,
            original: current_report.original,
            current_set: current_set,
            user_id: user_id
        })
    })

    .post('/eval', async(req, res) => {
        report_id = parseInt(req.body['current_set'])
        user_id = req.body['user_id']
        clinical_accuracy = req.body['clinical']
        grammatical_accuracy = req.body['grammatical']
        stylistic_quality = req.body['quality']
        edited_impression = req.body['edited_impression']
        edit_time = req.body['edit_time']
        await evaluation.create({ 
            user_id: user_id, 
            report_id: report_id,
            clinical_accuracy: clinical_accuracy,
            grammatical_accuracy: grammatical_accuracy,
            stylistic_quality: stylistic_quality,
            edited_impression: edited_impression,
            edit_time: edit_time
        })
        report_id += 1
        await user.update(
            { current_set: report_id },
            { where: { id: user_id } }
        )
        current_report = await report.findOne({ where: {id: report_id} }) 
        res.render('home', {
            exam: current_report.exam,
            clinical_history: current_report.clinical_history,
            comparison: current_report.comparison,
            findings: current_report.findings.replaceAll('\n', '<br/>'),
            original_impression: current_report.original_impression,
            predicted_impression: current_report.predicted_impression,
            original: current_report.original,
            current_set: report_id,
            user_id: user_id
        })
    })

    .listen(PORT, () => {
        console.log(`Server is running on port ${PORT}.`);
    })
