#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Wed Mar 26 23:49:45 2025
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code_control
from psychopy import visual,event,core
import os
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'RES501'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '',
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = [1440, 900]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/manastanavde/Downloads/PsychoPy/RES501/RES501.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_respStart') is None:
        # initialise key_respStart
        key_respStart = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_respStart',
        )
    if deviceManager.getDevice('key_resp_control') is None:
        # initialise key_resp_control
        key_resp_control = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_control',
        )
    if deviceManager.getDevice('key_resp_20') is None:
        # initialise key_resp_20
        key_resp_20 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_20',
        )
    if deviceManager.getDevice('key_resp_30') is None:
        # initialise key_resp_30
        key_resp_30 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_30',
        )
    if deviceManager.getDevice('key_resp_bundling') is None:
        # initialise key_resp_bundling
        key_resp_bundling = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_bundling',
        )
    if deviceManager.getDevice('key_respEnd') is None:
        # initialise key_respEnd
        key_respEnd = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_respEnd',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "WelcomeScreen" ---
    textWelcScreen = visual.TextStim(win=win, name='textWelcScreen',
        text='Welcome to our Experiment!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    textWelcMessage = visual.TextStim(win=win, name='textWelcMessage',
        text='You will be presented with several images that showcase subscription plans for various OTT platforms. \n\nPlease select a plan based on your preference. To do so, hit the numbers 1, 2, 3 or 4 based on which plan you want to select, read from the left. (For e.g. the first plan from the left would be 1, second from the left would be 2 etc.)\n\nPress SPACE to start',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_respStart = keyboard.Keyboard(deviceName='key_respStart')
    
    # --- Initialize components for Routine "trialControl" ---
    # Run 'Begin Experiment' code from code_control
    ### 1️⃣ BEGIN EXPERIMENT (Runs once at the start) ###
    import random 
    
    # Global Variables
    image_index = 0  # Tracks current trial index
    selected_plan = None  # Stores participant's choice
    
    # Define image folders for each condition
    image_paths = {
        "Control": os.path.abspath("Control_SubPlans/") + "/",
        "20": os.path.abspath("20_SubPlans/") + "/",
        "30": os.path.abspath("30_SubPlans/") + "/",
        "Bundling": os.path.abspath ("Bundling_SubPlans/") + "/"
        }
    
    # Define number of plans for each image
    plan_counts = {
        "Hotstar": 3,
        "Zee5": 3,
        "Netflix": 4,
        "AppleOne": 2,
        "SonyLiv": 3,
        "Crunchyroll": 3
    }
    
    # Generate image lists for each condition
    images_control = [f"{name}Control.jpg" for name in plan_counts.keys()]
    images_20 = [f"{name}20.jpg" for name in plan_counts.keys()]
    images_30 = [f"{name}30.jpg" for name in plan_counts.keys()]
    images_bundling = [f"{name}Bundling.jpg" for name in plan_counts.keys()]
    
    # Shuffle images for random presentation
    random.shuffle(images_control)
    random.shuffle(images_20)
    random.shuffle(images_30)
    random.shuffle(images_bundling)
    
    image_control = visual.ImageStim(
        win=win,
        name='image_control', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    key_resp_control = keyboard.Keyboard(deviceName='key_resp_control')
    
    # --- Initialize components for Routine "blank500" ---
    textblank500 = visual.TextStim(win=win, name='textblank500',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "blank500" ---
    textblank500 = visual.TextStim(win=win, name='textblank500',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial20" ---
    image_20 = visual.ImageStim(
        win=win,
        name='image_20', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    key_resp_20 = keyboard.Keyboard(deviceName='key_resp_20')
    
    # --- Initialize components for Routine "blank500" ---
    textblank500 = visual.TextStim(win=win, name='textblank500',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "blank500" ---
    textblank500 = visual.TextStim(win=win, name='textblank500',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial30" ---
    image_30 = visual.ImageStim(
        win=win,
        name='image_30', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    key_resp_30 = keyboard.Keyboard(deviceName='key_resp_30')
    
    # --- Initialize components for Routine "blank500" ---
    textblank500 = visual.TextStim(win=win, name='textblank500',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "blank500" ---
    textblank500 = visual.TextStim(win=win, name='textblank500',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trialBundling" ---
    image_bundling = visual.ImageStim(
        win=win,
        name='image_bundling', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    key_resp_bundling = keyboard.Keyboard(deviceName='key_resp_bundling')
    
    # --- Initialize components for Routine "blank500" ---
    textblank500 = visual.TextStim(win=win, name='textblank500',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "blank500" ---
    textblank500 = visual.TextStim(win=win, name='textblank500',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "EndScreen" ---
    textEndScreen = visual.TextStim(win=win, name='textEndScreen',
        text='Thanks for participating in the experiment! We greatly value your contribution to our study.\n\nPlease contact the experimenter about your completion.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_respEnd = keyboard.Keyboard(deviceName='key_respEnd')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "WelcomeScreen" ---
    # create an object to store info about Routine WelcomeScreen
    WelcomeScreen = data.Routine(
        name='WelcomeScreen',
        components=[textWelcScreen, textWelcMessage, key_respStart],
    )
    WelcomeScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_respStart
    key_respStart.keys = []
    key_respStart.rt = []
    _key_respStart_allKeys = []
    # store start times for WelcomeScreen
    WelcomeScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    WelcomeScreen.tStart = globalClock.getTime(format='float')
    WelcomeScreen.status = STARTED
    thisExp.addData('WelcomeScreen.started', WelcomeScreen.tStart)
    WelcomeScreen.maxDuration = None
    # keep track of which components have finished
    WelcomeScreenComponents = WelcomeScreen.components
    for thisComponent in WelcomeScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "WelcomeScreen" ---
    WelcomeScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textWelcScreen* updates
        
        # if textWelcScreen is starting this frame...
        if textWelcScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textWelcScreen.frameNStart = frameN  # exact frame index
            textWelcScreen.tStart = t  # local t and not account for scr refresh
            textWelcScreen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textWelcScreen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textWelcScreen.started')
            # update status
            textWelcScreen.status = STARTED
            textWelcScreen.setAutoDraw(True)
        
        # if textWelcScreen is active this frame...
        if textWelcScreen.status == STARTED:
            # update params
            pass
        
        # if textWelcScreen is stopping this frame...
        if textWelcScreen.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textWelcScreen.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                textWelcScreen.tStop = t  # not accounting for scr refresh
                textWelcScreen.tStopRefresh = tThisFlipGlobal  # on global time
                textWelcScreen.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textWelcScreen.stopped')
                # update status
                textWelcScreen.status = FINISHED
                textWelcScreen.setAutoDraw(False)
        
        # *textWelcMessage* updates
        
        # if textWelcMessage is starting this frame...
        if textWelcMessage.status == NOT_STARTED and tThisFlip >= 2.1-frameTolerance:
            # keep track of start time/frame for later
            textWelcMessage.frameNStart = frameN  # exact frame index
            textWelcMessage.tStart = t  # local t and not account for scr refresh
            textWelcMessage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textWelcMessage, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textWelcMessage.started')
            # update status
            textWelcMessage.status = STARTED
            textWelcMessage.setAutoDraw(True)
        
        # if textWelcMessage is active this frame...
        if textWelcMessage.status == STARTED:
            # update params
            pass
        
        # *key_respStart* updates
        waitOnFlip = False
        
        # if key_respStart is starting this frame...
        if key_respStart.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            key_respStart.frameNStart = frameN  # exact frame index
            key_respStart.tStart = t  # local t and not account for scr refresh
            key_respStart.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_respStart, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_respStart.started')
            # update status
            key_respStart.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_respStart.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_respStart.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_respStart.status == STARTED and not waitOnFlip:
            theseKeys = key_respStart.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_respStart_allKeys.extend(theseKeys)
            if len(_key_respStart_allKeys):
                key_respStart.keys = _key_respStart_allKeys[-1].name  # just the last key pressed
                key_respStart.rt = _key_respStart_allKeys[-1].rt
                key_respStart.duration = _key_respStart_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            WelcomeScreen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in WelcomeScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "WelcomeScreen" ---
    for thisComponent in WelcomeScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for WelcomeScreen
    WelcomeScreen.tStop = globalClock.getTime(format='float')
    WelcomeScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('WelcomeScreen.stopped', WelcomeScreen.tStop)
    # check responses
    if key_respStart.keys in ['', [], None]:  # No response was made
        key_respStart.keys = None
    thisExp.addData('key_respStart.keys',key_respStart.keys)
    if key_respStart.keys != None:  # we had a response
        thisExp.addData('key_respStart.rt', key_respStart.rt)
        thisExp.addData('key_respStart.duration', key_respStart.duration)
    thisExp.nextEntry()
    # the Routine "WelcomeScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trialsControl = data.TrialHandler2(
        name='trialsControl',
        nReps=6.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trialsControl)  # add the loop to the experiment
    thisTrialsControl = trialsControl.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrialsControl.rgb)
    if thisTrialsControl != None:
        for paramName in thisTrialsControl:
            globals()[paramName] = thisTrialsControl[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrialsControl in trialsControl:
        currentLoop = trialsControl
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrialsControl.rgb)
        if thisTrialsControl != None:
            for paramName in thisTrialsControl:
                globals()[paramName] = thisTrialsControl[paramName]
        
        # --- Prepare to start Routine "trialControl" ---
        # create an object to store info about Routine trialControl
        trialControl = data.Routine(
            name='trialControl',
            components=[image_control, key_resp_control],
        )
        trialControl.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_control
        ### 2️⃣ BEGIN ROUTINE (Runs at the start of each trial) ###
        
        # Determine the block type (Control, 20% Discount, or 30% Discount)
        block_type = "Control"  # Modify dynamically based on block assignment
        
        # Set image list and path based on block type
        if block_type == "Control":
            image_list = images_control
            image_path = image_paths["Control"]
        elif block_type == "20":
            image_list = images_20
            image_path = image_paths["20"]
        else:
            image_list = images_30
            image_path = image_paths["30"]
        
        # Load the current image
        if image_index < len(image_list):
            image_file_control = image_list[image_index]  
            current_image_control = image_path + image_file_control  # Full image path
        
            # Ensure the file exists
            if not os.path.exists(current_image_control):
                print(f"❌ ERROR: Missing image: {current_image_control}")
            else:
                print(f"✅ Displaying Image: {current_image_control}")
        
            # Get the number of plans for this image
            image_base_name = image_file_control.replace("Control.jpg", "").replace("20.jpg", "").replace("30.jpg", "")
            num_plans_control = plan_counts.get(image_base_name, 0)
        else:
            current_image_control = None
            num_plans_control = 0
        
        # Generate valid keys based on number of plans
        valid_keys_control = [str(i + 1) for i in range(num_plans_control)]
        valid_keys_str_control = valid_keys_control  # Store as list for PsychoPy keyboard component
        
        # Debugging output
        print(f"📝 Image: {image_file_control}, Plans Available: {num_plans_control}")
        print(f"🎯 Valid Keys: {valid_keys_str_control}")
        
        image_control.setSize([1,1])
        image_control.setImage(current_image_control)
        # create starting attributes for key_resp_control
        key_resp_control.keys = []
        key_resp_control.rt = []
        _key_resp_control_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'valid_keys_str_control' in globals():
            valid_keys_str_control = globals()['valid_keys_str_control']
        # store start times for trialControl
        trialControl.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trialControl.tStart = globalClock.getTime(format='float')
        trialControl.status = STARTED
        thisExp.addData('trialControl.started', trialControl.tStart)
        trialControl.maxDuration = None
        # keep track of which components have finished
        trialControlComponents = trialControl.components
        for thisComponent in trialControl.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trialControl" ---
        # if trial has changed, end Routine now
        if isinstance(trialsControl, data.TrialHandler2) and thisTrialsControl.thisN != trialsControl.thisTrial.thisN:
            continueRoutine = False
        trialControl.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_control* updates
            
            # if image_control is starting this frame...
            if image_control.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_control.frameNStart = frameN  # exact frame index
                image_control.tStart = t  # local t and not account for scr refresh
                image_control.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_control, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_control.started')
                # update status
                image_control.status = STARTED
                image_control.setAutoDraw(True)
            
            # if image_control is active this frame...
            if image_control.status == STARTED:
                # update params
                pass
            
            # *key_resp_control* updates
            waitOnFlip = False
            
            # if key_resp_control is starting this frame...
            if key_resp_control.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                key_resp_control.frameNStart = frameN  # exact frame index
                key_resp_control.tStart = t  # local t and not account for scr refresh
                key_resp_control.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_control, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_control.started')
                # update status
                key_resp_control.status = STARTED
                # allowed keys looks like a variable named `valid_keys_str_control`
                if not type(valid_keys_str_control) in [list, tuple, np.ndarray]:
                    if not isinstance(valid_keys_str_control, str):
                        valid_keys_str_control = str(valid_keys_str_control)
                    elif not ',' in valid_keys_str_control:
                        valid_keys_str_control = (valid_keys_str_control,)
                    else:
                        valid_keys_str_control = eval(valid_keys_str_control)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_control.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_control.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_control.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_control.getKeys(keyList=list(valid_keys_str_control), ignoreKeys=["escape"], waitRelease=False)
                _key_resp_control_allKeys.extend(theseKeys)
                if len(_key_resp_control_allKeys):
                    key_resp_control.keys = _key_resp_control_allKeys[-1].name  # just the last key pressed
                    key_resp_control.rt = _key_resp_control_allKeys[-1].rt
                    key_resp_control.duration = _key_resp_control_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trialControl.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialControl.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trialControl" ---
        for thisComponent in trialControl.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trialControl
        trialControl.tStop = globalClock.getTime(format='float')
        trialControl.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trialControl.stopped', trialControl.tStop)
        # Run 'End Routine' code from code_control
        ### 3️⃣ END ROUTINE (Runs after response is made) ###
        
        # Check if a key was pressed
        keys = key_resp_control.keys  # Ensure `key_resp` matches the Keyboard Component name
        
        if keys:
            selected_key_control = keys[0]  # Get the first key pressed
            selected_plan_control = f"Plan {selected_key_control}"
            print(f"✅ Selected: {selected_plan_control}")
        
            # End the routine when a valid key is pressed
            continueRoutine = False  
        else:
            selected_plan_control = "None"  # No key was pressed
            print("❌ No selection made.")
        
        # Store response in PsychoPy's data file
        thisExp.addData("Image", image_file_control)
        thisExp.addData("Selected_Plan", selected_plan_control)
        
        # Move to the next image
        image_index += 1
        if image_index >= len(image_list):  
            trialsControl.finished = True  # End the loop when all images are displayed
            image_index = 0 #Reset variable
            selected_plan = None #Reset variable
        
        # Debugging Output
        print(f"📜 Stored Plan: {selected_plan_control}")
        print(f"🔄 Moving to Image Index: {image_index}")
        
        # check responses
        if key_resp_control.keys in ['', [], None]:  # No response was made
            key_resp_control.keys = None
        trialsControl.addData('key_resp_control.keys',key_resp_control.keys)
        if key_resp_control.keys != None:  # we had a response
            trialsControl.addData('key_resp_control.rt', key_resp_control.rt)
            trialsControl.addData('key_resp_control.duration', key_resp_control.duration)
        # the Routine "trialControl" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "blank500" ---
        # create an object to store info about Routine blank500
        blank500 = data.Routine(
            name='blank500',
            components=[textblank500],
        )
        blank500.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank500
        blank500.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank500.tStart = globalClock.getTime(format='float')
        blank500.status = STARTED
        thisExp.addData('blank500.started', blank500.tStart)
        blank500.maxDuration = None
        # keep track of which components have finished
        blank500Components = blank500.components
        for thisComponent in blank500.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank500" ---
        # if trial has changed, end Routine now
        if isinstance(trialsControl, data.TrialHandler2) and thisTrialsControl.thisN != trialsControl.thisTrial.thisN:
            continueRoutine = False
        blank500.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textblank500* updates
            
            # if textblank500 is starting this frame...
            if textblank500.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textblank500.frameNStart = frameN  # exact frame index
                textblank500.tStart = t  # local t and not account for scr refresh
                textblank500.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textblank500, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textblank500.started')
                # update status
                textblank500.status = STARTED
                textblank500.setAutoDraw(True)
            
            # if textblank500 is active this frame...
            if textblank500.status == STARTED:
                # update params
                pass
            
            # if textblank500 is stopping this frame...
            if textblank500.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textblank500.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    textblank500.tStop = t  # not accounting for scr refresh
                    textblank500.tStopRefresh = tThisFlipGlobal  # on global time
                    textblank500.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textblank500.stopped')
                    # update status
                    textblank500.status = FINISHED
                    textblank500.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank500.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank500.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank500" ---
        for thisComponent in blank500.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank500
        blank500.tStop = globalClock.getTime(format='float')
        blank500.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank500.stopped', blank500.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank500.maxDurationReached:
            routineTimer.addTime(-blank500.maxDuration)
        elif blank500.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        thisExp.nextEntry()
        
    # completed 6.0 repeats of 'trialsControl'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if trialsControl.trialList in ([], [None], None):
        params = []
    else:
        params = trialsControl.trialList[0].keys()
    # save data for this loop
    trialsControl.saveAsExcel(filename + '.xlsx', sheetName='trialsControl',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "blank500" ---
    # create an object to store info about Routine blank500
    blank500 = data.Routine(
        name='blank500',
        components=[textblank500],
    )
    blank500.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for blank500
    blank500.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    blank500.tStart = globalClock.getTime(format='float')
    blank500.status = STARTED
    thisExp.addData('blank500.started', blank500.tStart)
    blank500.maxDuration = None
    # keep track of which components have finished
    blank500Components = blank500.components
    for thisComponent in blank500.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "blank500" ---
    blank500.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textblank500* updates
        
        # if textblank500 is starting this frame...
        if textblank500.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textblank500.frameNStart = frameN  # exact frame index
            textblank500.tStart = t  # local t and not account for scr refresh
            textblank500.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textblank500, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textblank500.started')
            # update status
            textblank500.status = STARTED
            textblank500.setAutoDraw(True)
        
        # if textblank500 is active this frame...
        if textblank500.status == STARTED:
            # update params
            pass
        
        # if textblank500 is stopping this frame...
        if textblank500.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textblank500.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                textblank500.tStop = t  # not accounting for scr refresh
                textblank500.tStopRefresh = tThisFlipGlobal  # on global time
                textblank500.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textblank500.stopped')
                # update status
                textblank500.status = FINISHED
                textblank500.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            blank500.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in blank500.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "blank500" ---
    for thisComponent in blank500.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for blank500
    blank500.tStop = globalClock.getTime(format='float')
    blank500.tStopRefresh = tThisFlipGlobal
    thisExp.addData('blank500.stopped', blank500.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if blank500.maxDurationReached:
        routineTimer.addTime(-blank500.maxDuration)
    elif blank500.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.500000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trials20 = data.TrialHandler2(
        name='trials20',
        nReps=6.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials20)  # add the loop to the experiment
    thisTrials20 = trials20.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials20.rgb)
    if thisTrials20 != None:
        for paramName in thisTrials20:
            globals()[paramName] = thisTrials20[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials20 in trials20:
        currentLoop = trials20
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials20.rgb)
        if thisTrials20 != None:
            for paramName in thisTrials20:
                globals()[paramName] = thisTrials20[paramName]
        
        # --- Prepare to start Routine "trial20" ---
        # create an object to store info about Routine trial20
        trial20 = data.Routine(
            name='trial20',
            components=[image_20, key_resp_20],
        )
        trial20.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_20
        ### BEGIN ROUTINE for 20% Discount ###
        block_type = "20"
        
        if block_type == "20":
            image_list_20 = images_20
            image_path_20 = image_paths["20"]
        
        if image_index < len(image_list_20):
            image_file_20 = image_list_20[image_index]
            current_image_20 = image_path_20 + image_file_20
        
            if not os.path.exists(current_image_20):
                print(f"❌ ERROR: Missing image: {current_image_20}")
            else:
                print(f"✅ Displaying Image: {current_image_20}")
        
            image_base_name = image_file_20.replace("20.jpg", "")
            num_plans_20 = plan_counts.get(image_base_name, 0)
        else:
            current_image_20 = None
            num_plans_20 = 0
        
        valid_keys_20 = [str(i + 1) for i in range(num_plans_20)]
        valid_keys_str_20 = valid_keys_20
        
        image_20.setSize([1,1])
        image_20.setImage(current_image_20)
        # create starting attributes for key_resp_20
        key_resp_20.keys = []
        key_resp_20.rt = []
        _key_resp_20_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'valid_keys_str_20' in globals():
            valid_keys_str_20 = globals()['valid_keys_str_20']
        # store start times for trial20
        trial20.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial20.tStart = globalClock.getTime(format='float')
        trial20.status = STARTED
        thisExp.addData('trial20.started', trial20.tStart)
        trial20.maxDuration = None
        # keep track of which components have finished
        trial20Components = trial20.components
        for thisComponent in trial20.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial20" ---
        # if trial has changed, end Routine now
        if isinstance(trials20, data.TrialHandler2) and thisTrials20.thisN != trials20.thisTrial.thisN:
            continueRoutine = False
        trial20.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_20* updates
            
            # if image_20 is starting this frame...
            if image_20.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_20.frameNStart = frameN  # exact frame index
                image_20.tStart = t  # local t and not account for scr refresh
                image_20.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_20, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_20.started')
                # update status
                image_20.status = STARTED
                image_20.setAutoDraw(True)
            
            # if image_20 is active this frame...
            if image_20.status == STARTED:
                # update params
                pass
            
            # *key_resp_20* updates
            waitOnFlip = False
            
            # if key_resp_20 is starting this frame...
            if key_resp_20.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                key_resp_20.frameNStart = frameN  # exact frame index
                key_resp_20.tStart = t  # local t and not account for scr refresh
                key_resp_20.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_20, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_20.started')
                # update status
                key_resp_20.status = STARTED
                # allowed keys looks like a variable named `valid_keys_str_20`
                if not type(valid_keys_str_20) in [list, tuple, np.ndarray]:
                    if not isinstance(valid_keys_str_20, str):
                        valid_keys_str_20 = str(valid_keys_str_20)
                    elif not ',' in valid_keys_str_20:
                        valid_keys_str_20 = (valid_keys_str_20,)
                    else:
                        valid_keys_str_20 = eval(valid_keys_str_20)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_20.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_20.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_20.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_20.getKeys(keyList=list(valid_keys_str_20), ignoreKeys=["escape"], waitRelease=False)
                _key_resp_20_allKeys.extend(theseKeys)
                if len(_key_resp_20_allKeys):
                    key_resp_20.keys = _key_resp_20_allKeys[-1].name  # just the last key pressed
                    key_resp_20.rt = _key_resp_20_allKeys[-1].rt
                    key_resp_20.duration = _key_resp_20_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial20.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial20.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial20" ---
        for thisComponent in trial20.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial20
        trial20.tStop = globalClock.getTime(format='float')
        trial20.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial20.stopped', trial20.tStop)
        # Run 'End Routine' code from code_20
        ### END ROUTINE for 20% Discount ###
        keys_20 = key_resp_20.keys
        
        if keys_20:
            selected_key_20 = keys_20[0]
            selected_plan_20 = f"Plan {selected_key_20}"
            print(f"✅ Selected: {selected_plan_20}")
        
            continueRoutine = False  
        else:
            selected_plan_20 = "None"
            print("❌ No selection made.")
        
        thisExp.addData("Image_20", image_file_20)
        thisExp.addData("Selected_Plan_20", selected_plan_20)
        
        image_index += 1
        if image_index >= len(image_list_20):  
            trials20.finished = True  
            image_index = 0 #Reset variable
            selected_plan = None #Reset variable
        
        print(f"📜 Stored Plan: {selected_plan_20}")
        print(f"🔄 Moving to Image Index: {image_index}")
        
        # check responses
        if key_resp_20.keys in ['', [], None]:  # No response was made
            key_resp_20.keys = None
        trials20.addData('key_resp_20.keys',key_resp_20.keys)
        if key_resp_20.keys != None:  # we had a response
            trials20.addData('key_resp_20.rt', key_resp_20.rt)
            trials20.addData('key_resp_20.duration', key_resp_20.duration)
        # the Routine "trial20" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "blank500" ---
        # create an object to store info about Routine blank500
        blank500 = data.Routine(
            name='blank500',
            components=[textblank500],
        )
        blank500.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank500
        blank500.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank500.tStart = globalClock.getTime(format='float')
        blank500.status = STARTED
        thisExp.addData('blank500.started', blank500.tStart)
        blank500.maxDuration = None
        # keep track of which components have finished
        blank500Components = blank500.components
        for thisComponent in blank500.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank500" ---
        # if trial has changed, end Routine now
        if isinstance(trials20, data.TrialHandler2) and thisTrials20.thisN != trials20.thisTrial.thisN:
            continueRoutine = False
        blank500.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textblank500* updates
            
            # if textblank500 is starting this frame...
            if textblank500.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textblank500.frameNStart = frameN  # exact frame index
                textblank500.tStart = t  # local t and not account for scr refresh
                textblank500.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textblank500, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textblank500.started')
                # update status
                textblank500.status = STARTED
                textblank500.setAutoDraw(True)
            
            # if textblank500 is active this frame...
            if textblank500.status == STARTED:
                # update params
                pass
            
            # if textblank500 is stopping this frame...
            if textblank500.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textblank500.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    textblank500.tStop = t  # not accounting for scr refresh
                    textblank500.tStopRefresh = tThisFlipGlobal  # on global time
                    textblank500.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textblank500.stopped')
                    # update status
                    textblank500.status = FINISHED
                    textblank500.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank500.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank500.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank500" ---
        for thisComponent in blank500.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank500
        blank500.tStop = globalClock.getTime(format='float')
        blank500.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank500.stopped', blank500.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank500.maxDurationReached:
            routineTimer.addTime(-blank500.maxDuration)
        elif blank500.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        thisExp.nextEntry()
        
    # completed 6.0 repeats of 'trials20'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if trials20.trialList in ([], [None], None):
        params = []
    else:
        params = trials20.trialList[0].keys()
    # save data for this loop
    trials20.saveAsExcel(filename + '.xlsx', sheetName='trials20',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "blank500" ---
    # create an object to store info about Routine blank500
    blank500 = data.Routine(
        name='blank500',
        components=[textblank500],
    )
    blank500.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for blank500
    blank500.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    blank500.tStart = globalClock.getTime(format='float')
    blank500.status = STARTED
    thisExp.addData('blank500.started', blank500.tStart)
    blank500.maxDuration = None
    # keep track of which components have finished
    blank500Components = blank500.components
    for thisComponent in blank500.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "blank500" ---
    blank500.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textblank500* updates
        
        # if textblank500 is starting this frame...
        if textblank500.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textblank500.frameNStart = frameN  # exact frame index
            textblank500.tStart = t  # local t and not account for scr refresh
            textblank500.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textblank500, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textblank500.started')
            # update status
            textblank500.status = STARTED
            textblank500.setAutoDraw(True)
        
        # if textblank500 is active this frame...
        if textblank500.status == STARTED:
            # update params
            pass
        
        # if textblank500 is stopping this frame...
        if textblank500.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textblank500.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                textblank500.tStop = t  # not accounting for scr refresh
                textblank500.tStopRefresh = tThisFlipGlobal  # on global time
                textblank500.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textblank500.stopped')
                # update status
                textblank500.status = FINISHED
                textblank500.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            blank500.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in blank500.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "blank500" ---
    for thisComponent in blank500.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for blank500
    blank500.tStop = globalClock.getTime(format='float')
    blank500.tStopRefresh = tThisFlipGlobal
    thisExp.addData('blank500.stopped', blank500.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if blank500.maxDurationReached:
        routineTimer.addTime(-blank500.maxDuration)
    elif blank500.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.500000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trials30 = data.TrialHandler2(
        name='trials30',
        nReps=6.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials30)  # add the loop to the experiment
    thisTrials30 = trials30.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials30.rgb)
    if thisTrials30 != None:
        for paramName in thisTrials30:
            globals()[paramName] = thisTrials30[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials30 in trials30:
        currentLoop = trials30
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials30.rgb)
        if thisTrials30 != None:
            for paramName in thisTrials30:
                globals()[paramName] = thisTrials30[paramName]
        
        # --- Prepare to start Routine "trial30" ---
        # create an object to store info about Routine trial30
        trial30 = data.Routine(
            name='trial30',
            components=[image_30, key_resp_30],
        )
        trial30.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_30
        ### BEGIN ROUTINE for 30% Discount ###
        block_type = "30"
        
        if block_type == "30":
            image_list_30 = images_30
            image_path_30 = image_paths["30"]
        
        if image_index < len(image_list_30):
            image_file_30 = image_list_30[image_index]
            current_image_30 = image_path_30 + image_file_30
        
            if not os.path.exists(current_image_30):
                print(f"❌ ERROR: Missing image: {current_image_30}")
            else:
                print(f"✅ Displaying Image: {current_image_30}")
        
            image_base_name = image_file_30.replace("30.jpg", "")
            num_plans_30 = plan_counts.get(image_base_name, 0)
        else:
            current_image_30 = None
            num_plans_30 = 0
        
        valid_keys_30 = [str(i + 1) for i in range(num_plans_30)]
        valid_keys_str_30 = valid_keys_30
        
        image_30.setSize([1,1])
        image_30.setImage(current_image_30)
        # create starting attributes for key_resp_30
        key_resp_30.keys = []
        key_resp_30.rt = []
        _key_resp_30_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'valid_keys_str_30' in globals():
            valid_keys_str_30 = globals()['valid_keys_str_30']
        # store start times for trial30
        trial30.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial30.tStart = globalClock.getTime(format='float')
        trial30.status = STARTED
        thisExp.addData('trial30.started', trial30.tStart)
        trial30.maxDuration = None
        # keep track of which components have finished
        trial30Components = trial30.components
        for thisComponent in trial30.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial30" ---
        # if trial has changed, end Routine now
        if isinstance(trials30, data.TrialHandler2) and thisTrials30.thisN != trials30.thisTrial.thisN:
            continueRoutine = False
        trial30.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_30* updates
            
            # if image_30 is starting this frame...
            if image_30.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_30.frameNStart = frameN  # exact frame index
                image_30.tStart = t  # local t and not account for scr refresh
                image_30.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_30, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_30.started')
                # update status
                image_30.status = STARTED
                image_30.setAutoDraw(True)
            
            # if image_30 is active this frame...
            if image_30.status == STARTED:
                # update params
                pass
            
            # *key_resp_30* updates
            waitOnFlip = False
            
            # if key_resp_30 is starting this frame...
            if key_resp_30.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                key_resp_30.frameNStart = frameN  # exact frame index
                key_resp_30.tStart = t  # local t and not account for scr refresh
                key_resp_30.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_30, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_30.started')
                # update status
                key_resp_30.status = STARTED
                # allowed keys looks like a variable named `valid_keys_str_30`
                if not type(valid_keys_str_30) in [list, tuple, np.ndarray]:
                    if not isinstance(valid_keys_str_30, str):
                        valid_keys_str_30 = str(valid_keys_str_30)
                    elif not ',' in valid_keys_str_30:
                        valid_keys_str_30 = (valid_keys_str_30,)
                    else:
                        valid_keys_str_30 = eval(valid_keys_str_30)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_30.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_30.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_30.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_30.getKeys(keyList=list(valid_keys_str_30), ignoreKeys=["escape"], waitRelease=False)
                _key_resp_30_allKeys.extend(theseKeys)
                if len(_key_resp_30_allKeys):
                    key_resp_30.keys = _key_resp_30_allKeys[-1].name  # just the last key pressed
                    key_resp_30.rt = _key_resp_30_allKeys[-1].rt
                    key_resp_30.duration = _key_resp_30_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial30.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial30.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial30" ---
        for thisComponent in trial30.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial30
        trial30.tStop = globalClock.getTime(format='float')
        trial30.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial30.stopped', trial30.tStop)
        # Run 'End Routine' code from code_30
        ### END ROUTINE for 30% Discount ###
        keys_30 = key_resp_30.keys
        
        if keys_30:
            selected_key_30 = keys_30[0]
            selected_plan_30 = f"Plan {selected_key_30}"
            print(f"✅ Selected: {selected_plan_30}")
        
            continueRoutine = False  
        else:
            selected_plan_30 = "None"
            print("❌ No selection made.")
        
        thisExp.addData("Image_30", image_file_30)
        thisExp.addData("Selected_Plan_30", selected_plan_30)
        
        image_index += 1
        if image_index >= len(image_list_30):  
            trials30.finished = True
            image_index = 0 #Reset variable
            selected_plan = None #Reset variable
        
        print(f"📜 Stored Plan: {selected_plan_30}")
        print(f"🔄 Moving to Image Index: {image_index}")
        
        # check responses
        if key_resp_30.keys in ['', [], None]:  # No response was made
            key_resp_30.keys = None
        trials30.addData('key_resp_30.keys',key_resp_30.keys)
        if key_resp_30.keys != None:  # we had a response
            trials30.addData('key_resp_30.rt', key_resp_30.rt)
            trials30.addData('key_resp_30.duration', key_resp_30.duration)
        # the Routine "trial30" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "blank500" ---
        # create an object to store info about Routine blank500
        blank500 = data.Routine(
            name='blank500',
            components=[textblank500],
        )
        blank500.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank500
        blank500.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank500.tStart = globalClock.getTime(format='float')
        blank500.status = STARTED
        thisExp.addData('blank500.started', blank500.tStart)
        blank500.maxDuration = None
        # keep track of which components have finished
        blank500Components = blank500.components
        for thisComponent in blank500.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank500" ---
        # if trial has changed, end Routine now
        if isinstance(trials30, data.TrialHandler2) and thisTrials30.thisN != trials30.thisTrial.thisN:
            continueRoutine = False
        blank500.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textblank500* updates
            
            # if textblank500 is starting this frame...
            if textblank500.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textblank500.frameNStart = frameN  # exact frame index
                textblank500.tStart = t  # local t and not account for scr refresh
                textblank500.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textblank500, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textblank500.started')
                # update status
                textblank500.status = STARTED
                textblank500.setAutoDraw(True)
            
            # if textblank500 is active this frame...
            if textblank500.status == STARTED:
                # update params
                pass
            
            # if textblank500 is stopping this frame...
            if textblank500.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textblank500.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    textblank500.tStop = t  # not accounting for scr refresh
                    textblank500.tStopRefresh = tThisFlipGlobal  # on global time
                    textblank500.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textblank500.stopped')
                    # update status
                    textblank500.status = FINISHED
                    textblank500.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank500.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank500.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank500" ---
        for thisComponent in blank500.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank500
        blank500.tStop = globalClock.getTime(format='float')
        blank500.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank500.stopped', blank500.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank500.maxDurationReached:
            routineTimer.addTime(-blank500.maxDuration)
        elif blank500.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        thisExp.nextEntry()
        
    # completed 6.0 repeats of 'trials30'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if trials30.trialList in ([], [None], None):
        params = []
    else:
        params = trials30.trialList[0].keys()
    # save data for this loop
    trials30.saveAsExcel(filename + '.xlsx', sheetName='trials30',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "blank500" ---
    # create an object to store info about Routine blank500
    blank500 = data.Routine(
        name='blank500',
        components=[textblank500],
    )
    blank500.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for blank500
    blank500.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    blank500.tStart = globalClock.getTime(format='float')
    blank500.status = STARTED
    thisExp.addData('blank500.started', blank500.tStart)
    blank500.maxDuration = None
    # keep track of which components have finished
    blank500Components = blank500.components
    for thisComponent in blank500.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "blank500" ---
    blank500.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textblank500* updates
        
        # if textblank500 is starting this frame...
        if textblank500.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textblank500.frameNStart = frameN  # exact frame index
            textblank500.tStart = t  # local t and not account for scr refresh
            textblank500.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textblank500, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textblank500.started')
            # update status
            textblank500.status = STARTED
            textblank500.setAutoDraw(True)
        
        # if textblank500 is active this frame...
        if textblank500.status == STARTED:
            # update params
            pass
        
        # if textblank500 is stopping this frame...
        if textblank500.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textblank500.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                textblank500.tStop = t  # not accounting for scr refresh
                textblank500.tStopRefresh = tThisFlipGlobal  # on global time
                textblank500.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textblank500.stopped')
                # update status
                textblank500.status = FINISHED
                textblank500.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            blank500.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in blank500.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "blank500" ---
    for thisComponent in blank500.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for blank500
    blank500.tStop = globalClock.getTime(format='float')
    blank500.tStopRefresh = tThisFlipGlobal
    thisExp.addData('blank500.stopped', blank500.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if blank500.maxDurationReached:
        routineTimer.addTime(-blank500.maxDuration)
    elif blank500.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.500000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trialsBundling = data.TrialHandler2(
        name='trialsBundling',
        nReps=6.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trialsBundling)  # add the loop to the experiment
    thisTrialsBundling = trialsBundling.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrialsBundling.rgb)
    if thisTrialsBundling != None:
        for paramName in thisTrialsBundling:
            globals()[paramName] = thisTrialsBundling[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrialsBundling in trialsBundling:
        currentLoop = trialsBundling
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrialsBundling.rgb)
        if thisTrialsBundling != None:
            for paramName in thisTrialsBundling:
                globals()[paramName] = thisTrialsBundling[paramName]
        
        # --- Prepare to start Routine "trialBundling" ---
        # create an object to store info about Routine trialBundling
        trialBundling = data.Routine(
            name='trialBundling',
            components=[image_bundling, key_resp_bundling],
        )
        trialBundling.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_bundling
        ### BEGIN ROUTINE for Bundling ###
        block_type = "Bundling"
        
        if block_type == "Bundling":
            image_list_bundling = images_bundling
            image_path_bundling = image_paths["Bundling"]
        
        if image_index < len(image_list_bundling):
            image_file_bundling = image_list_bundling[image_index]
            current_image_bundling = image_path_bundling + image_file_bundling
        
            if not os.path.exists(current_image_bundling):
                print(f"❌ ERROR: Missing image: {current_image_bundling}")
            else:
                print(f"✅ Displaying Image: {current_image_bundling}")
        
            image_base_name = image_file_bundling.replace("Bundling.jpg", "")
            num_plans_bundling = plan_counts.get(image_base_name, 0)
        else:
            current_image_bundling = None
            num_plans_bundling = 0
        
        valid_keys_bundling = [str(i + 1) for i in range(num_plans_bundling)]
        valid_keys_str_bundling = valid_keys_bundling
        
        image_bundling.setSize([1,0.85])
        image_bundling.setImage(current_image_bundling)
        # create starting attributes for key_resp_bundling
        key_resp_bundling.keys = []
        key_resp_bundling.rt = []
        _key_resp_bundling_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'valid_keys_str_bundling' in globals():
            valid_keys_str_bundling = globals()['valid_keys_str_bundling']
        # store start times for trialBundling
        trialBundling.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trialBundling.tStart = globalClock.getTime(format='float')
        trialBundling.status = STARTED
        thisExp.addData('trialBundling.started', trialBundling.tStart)
        trialBundling.maxDuration = None
        # keep track of which components have finished
        trialBundlingComponents = trialBundling.components
        for thisComponent in trialBundling.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trialBundling" ---
        # if trial has changed, end Routine now
        if isinstance(trialsBundling, data.TrialHandler2) and thisTrialsBundling.thisN != trialsBundling.thisTrial.thisN:
            continueRoutine = False
        trialBundling.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_bundling* updates
            
            # if image_bundling is starting this frame...
            if image_bundling.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_bundling.frameNStart = frameN  # exact frame index
                image_bundling.tStart = t  # local t and not account for scr refresh
                image_bundling.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_bundling, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_bundling.started')
                # update status
                image_bundling.status = STARTED
                image_bundling.setAutoDraw(True)
            
            # if image_bundling is active this frame...
            if image_bundling.status == STARTED:
                # update params
                pass
            
            # *key_resp_bundling* updates
            waitOnFlip = False
            
            # if key_resp_bundling is starting this frame...
            if key_resp_bundling.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                key_resp_bundling.frameNStart = frameN  # exact frame index
                key_resp_bundling.tStart = t  # local t and not account for scr refresh
                key_resp_bundling.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_bundling, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_bundling.started')
                # update status
                key_resp_bundling.status = STARTED
                # allowed keys looks like a variable named `valid_keys_str_bundling`
                if not type(valid_keys_str_bundling) in [list, tuple, np.ndarray]:
                    if not isinstance(valid_keys_str_bundling, str):
                        valid_keys_str_bundling = str(valid_keys_str_bundling)
                    elif not ',' in valid_keys_str_bundling:
                        valid_keys_str_bundling = (valid_keys_str_bundling,)
                    else:
                        valid_keys_str_bundling = eval(valid_keys_str_bundling)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_bundling.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_bundling.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_bundling.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_bundling.getKeys(keyList=list(valid_keys_str_bundling), ignoreKeys=["escape"], waitRelease=False)
                _key_resp_bundling_allKeys.extend(theseKeys)
                if len(_key_resp_bundling_allKeys):
                    key_resp_bundling.keys = _key_resp_bundling_allKeys[-1].name  # just the last key pressed
                    key_resp_bundling.rt = _key_resp_bundling_allKeys[-1].rt
                    key_resp_bundling.duration = _key_resp_bundling_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trialBundling.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialBundling.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trialBundling" ---
        for thisComponent in trialBundling.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trialBundling
        trialBundling.tStop = globalClock.getTime(format='float')
        trialBundling.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trialBundling.stopped', trialBundling.tStop)
        # Run 'End Routine' code from code_bundling
        ### END ROUTINE for Bundling ###
        if key_resp_bundling.keys:  # ✅ Corrected Variable Name
            keys_bundling = key_resp_bundling.keys  # ✅ Assign response keys correctly
            selected_key_bundling = keys_bundling[0]  # Get first key press
            selected_plan_bundling = f"Plan {selected_key_bundling}"
            print(f"✅ Selected: {selected_plan_bundling}")
        
            continueRoutine = False  # Ensure routine ends immediately  
        else:
            selected_plan_bundling = "None"
            print("❌ No selection made.")
        
        thisExp.addData("Image_Bundling", image_file_bundling)
        thisExp.addData("Selected_Plan_Bundling", selected_plan_bundling)
        
        # ✅ Ensure routine ends before incrementing image_index
        if not continueRoutine:
            image_index += 1
        
            # ✅ Ensure loop stops after the last image
            if image_index >= len(image_list_bundling):  
                print("✅ All Bundling trials completed! Ending experiment.")
                trialsBundling.finished = True  # ✅ Ensures loop stops only once
        
        print(f"📜 Stored Plan: {selected_plan_bundling}")
        print(f"🔄 Moving to Image Index: {image_index}")
        
        # check responses
        if key_resp_bundling.keys in ['', [], None]:  # No response was made
            key_resp_bundling.keys = None
        trialsBundling.addData('key_resp_bundling.keys',key_resp_bundling.keys)
        if key_resp_bundling.keys != None:  # we had a response
            trialsBundling.addData('key_resp_bundling.rt', key_resp_bundling.rt)
            trialsBundling.addData('key_resp_bundling.duration', key_resp_bundling.duration)
        # the Routine "trialBundling" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "blank500" ---
        # create an object to store info about Routine blank500
        blank500 = data.Routine(
            name='blank500',
            components=[textblank500],
        )
        blank500.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank500
        blank500.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank500.tStart = globalClock.getTime(format='float')
        blank500.status = STARTED
        thisExp.addData('blank500.started', blank500.tStart)
        blank500.maxDuration = None
        # keep track of which components have finished
        blank500Components = blank500.components
        for thisComponent in blank500.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank500" ---
        # if trial has changed, end Routine now
        if isinstance(trialsBundling, data.TrialHandler2) and thisTrialsBundling.thisN != trialsBundling.thisTrial.thisN:
            continueRoutine = False
        blank500.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textblank500* updates
            
            # if textblank500 is starting this frame...
            if textblank500.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textblank500.frameNStart = frameN  # exact frame index
                textblank500.tStart = t  # local t and not account for scr refresh
                textblank500.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textblank500, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textblank500.started')
                # update status
                textblank500.status = STARTED
                textblank500.setAutoDraw(True)
            
            # if textblank500 is active this frame...
            if textblank500.status == STARTED:
                # update params
                pass
            
            # if textblank500 is stopping this frame...
            if textblank500.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textblank500.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    textblank500.tStop = t  # not accounting for scr refresh
                    textblank500.tStopRefresh = tThisFlipGlobal  # on global time
                    textblank500.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textblank500.stopped')
                    # update status
                    textblank500.status = FINISHED
                    textblank500.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank500.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank500.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank500" ---
        for thisComponent in blank500.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank500
        blank500.tStop = globalClock.getTime(format='float')
        blank500.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank500.stopped', blank500.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank500.maxDurationReached:
            routineTimer.addTime(-blank500.maxDuration)
        elif blank500.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        thisExp.nextEntry()
        
    # completed 6.0 repeats of 'trialsBundling'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if trialsBundling.trialList in ([], [None], None):
        params = []
    else:
        params = trialsBundling.trialList[0].keys()
    # save data for this loop
    trialsBundling.saveAsExcel(filename + '.xlsx', sheetName='trialsBundling',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "blank500" ---
    # create an object to store info about Routine blank500
    blank500 = data.Routine(
        name='blank500',
        components=[textblank500],
    )
    blank500.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for blank500
    blank500.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    blank500.tStart = globalClock.getTime(format='float')
    blank500.status = STARTED
    thisExp.addData('blank500.started', blank500.tStart)
    blank500.maxDuration = None
    # keep track of which components have finished
    blank500Components = blank500.components
    for thisComponent in blank500.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "blank500" ---
    blank500.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textblank500* updates
        
        # if textblank500 is starting this frame...
        if textblank500.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textblank500.frameNStart = frameN  # exact frame index
            textblank500.tStart = t  # local t and not account for scr refresh
            textblank500.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textblank500, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textblank500.started')
            # update status
            textblank500.status = STARTED
            textblank500.setAutoDraw(True)
        
        # if textblank500 is active this frame...
        if textblank500.status == STARTED:
            # update params
            pass
        
        # if textblank500 is stopping this frame...
        if textblank500.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > textblank500.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                textblank500.tStop = t  # not accounting for scr refresh
                textblank500.tStopRefresh = tThisFlipGlobal  # on global time
                textblank500.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textblank500.stopped')
                # update status
                textblank500.status = FINISHED
                textblank500.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            blank500.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in blank500.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "blank500" ---
    for thisComponent in blank500.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for blank500
    blank500.tStop = globalClock.getTime(format='float')
    blank500.tStopRefresh = tThisFlipGlobal
    thisExp.addData('blank500.stopped', blank500.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if blank500.maxDurationReached:
        routineTimer.addTime(-blank500.maxDuration)
    elif blank500.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.500000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "EndScreen" ---
    # create an object to store info about Routine EndScreen
    EndScreen = data.Routine(
        name='EndScreen',
        components=[textEndScreen, key_respEnd],
    )
    EndScreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_respEnd
    key_respEnd.keys = []
    key_respEnd.rt = []
    _key_respEnd_allKeys = []
    # store start times for EndScreen
    EndScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    EndScreen.tStart = globalClock.getTime(format='float')
    EndScreen.status = STARTED
    thisExp.addData('EndScreen.started', EndScreen.tStart)
    EndScreen.maxDuration = None
    # keep track of which components have finished
    EndScreenComponents = EndScreen.components
    for thisComponent in EndScreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "EndScreen" ---
    EndScreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textEndScreen* updates
        
        # if textEndScreen is starting this frame...
        if textEndScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textEndScreen.frameNStart = frameN  # exact frame index
            textEndScreen.tStart = t  # local t and not account for scr refresh
            textEndScreen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textEndScreen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textEndScreen.started')
            # update status
            textEndScreen.status = STARTED
            textEndScreen.setAutoDraw(True)
        
        # if textEndScreen is active this frame...
        if textEndScreen.status == STARTED:
            # update params
            pass
        
        # *key_respEnd* updates
        waitOnFlip = False
        
        # if key_respEnd is starting this frame...
        if key_respEnd.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
            # keep track of start time/frame for later
            key_respEnd.frameNStart = frameN  # exact frame index
            key_respEnd.tStart = t  # local t and not account for scr refresh
            key_respEnd.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_respEnd, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_respEnd.started')
            # update status
            key_respEnd.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_respEnd.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_respEnd.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_respEnd.status == STARTED and not waitOnFlip:
            theseKeys = key_respEnd.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_respEnd_allKeys.extend(theseKeys)
            if len(_key_respEnd_allKeys):
                key_respEnd.keys = _key_respEnd_allKeys[-1].name  # just the last key pressed
                key_respEnd.rt = _key_respEnd_allKeys[-1].rt
                key_respEnd.duration = _key_respEnd_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            EndScreen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in EndScreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "EndScreen" ---
    for thisComponent in EndScreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for EndScreen
    EndScreen.tStop = globalClock.getTime(format='float')
    EndScreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('EndScreen.stopped', EndScreen.tStop)
    # check responses
    if key_respEnd.keys in ['', [], None]:  # No response was made
        key_respEnd.keys = None
    thisExp.addData('key_respEnd.keys',key_respEnd.keys)
    if key_respEnd.keys != None:  # we had a response
        thisExp.addData('key_respEnd.rt', key_respEnd.rt)
        thisExp.addData('key_respEnd.duration', key_respEnd.duration)
    thisExp.nextEntry()
    # the Routine "EndScreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
