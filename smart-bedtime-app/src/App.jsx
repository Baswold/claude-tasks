import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [workNightBedtime, setWorkNightBedtime] = useState('23:00')
  const [freeNightBedtime, setFreeNightBedtime] = useState('01:00')
  const [calendarConnected, setCalendarConnected] = useState(false)
  const [isInFlow, setIsInFlow] = useState(false)
  const [currentTime, setCurrentTime] = useState(new Date())
  const [showBedtimeAlert, setShowBedtimeAlert] = useState(false)
  const [hasWorkTomorrow, setHasWorkTomorrow] = useState(true)

  // Update current time every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  // Check if it's bedtime
  useEffect(() => {
    const targetBedtime = hasWorkTomorrow ? workNightBedtime : freeNightBedtime
    const [hours, minutes] = targetBedtime.split(':').map(Number)
    const now = new Date()

    if (now.getHours() === hours && now.getMinutes() === minutes && !isInFlow) {
      setShowBedtimeAlert(true)
      // Request notification permission and show notification
      if (Notification.permission === 'granted') {
        new Notification('Time for bed', {
          body: `It's ${targetBedtime}. Time to rest for tomorrow.`,
          icon: '🌙'
        })
      }
    }
  }, [currentTime, workNightBedtime, freeNightBedtime, hasWorkTomorrow, isInFlow])

  // Request notification permission on load
  useEffect(() => {
    if (Notification.permission === 'default') {
      Notification.requestPermission()
    }
  }, [])

  const connectCalendar = () => {
    // Simulate calendar connection
    setCalendarConnected(true)
    // In production, this would use Google Calendar API
    setTimeout(() => {
      // Simulate checking calendar
      const hasEvents = Math.random() > 0.5
      setHasWorkTomorrow(hasEvents)
    }, 1000)
  }

  const toggleFlowState = () => {
    setIsInFlow(!isInFlow)
  }

  const dismissAlert = () => {
    setShowBedtimeAlert(false)
  }

  const formatTime = (time) => {
    return time.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    })
  }

  const getTargetBedtime = () => {
    return hasWorkTomorrow ? workNightBedtime : freeNightBedtime
  }

  return (
    <div className="app">
      {showBedtimeAlert && !isInFlow && (
        <div className="bedtime-alert">
          <div className="alert-content">
            <div className="alert-icon">🌙</div>
            <h2>Time for bed</h2>
            <p>It's {getTargetBedtime()}. Time to rest for tomorrow.</p>
            <button onClick={dismissAlert} className="dismiss-btn">
              Got it
            </button>
          </div>
        </div>
      )}

      <div className="container">
        <header>
          <div className="logo">
            <span className="moon-icon">🌙</span>
          </div>
          <h1>FlowBed</h1>
          <p className="tagline">Smart bedtime for developers in flow</p>
        </header>

        <div className="current-time-display">
          <div className="time">{formatTime(currentTime)}</div>
          <div className="date">{currentTime.toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
          })}</div>
        </div>

        <div className="status-cards">
          <div className={`status-card ${isInFlow ? 'active' : ''}`}>
            <div className="status-indicator">
              <div className={`pulse ${isInFlow ? 'pulsing' : ''}`}></div>
            </div>
            <h3>Flow State</h3>
            <p>{isInFlow ? 'In deep focus' : 'Not in flow'}</p>
            <button onClick={toggleFlowState} className="toggle-btn">
              {isInFlow ? 'Exit flow' : 'Enter flow'}
            </button>
          </div>

          <div className={`status-card ${calendarConnected ? 'connected' : ''}`}>
            <div className="status-indicator">
              <div className={`pulse ${calendarConnected ? 'connected-pulse' : ''}`}></div>
            </div>
            <h3>Calendar</h3>
            <p>{calendarConnected ? 'Connected' : 'Not connected'}</p>
            {!calendarConnected && (
              <button onClick={connectCalendar} className="connect-btn">
                Connect
              </button>
            )}
            {calendarConnected && (
              <div className="calendar-status">
                {hasWorkTomorrow ? '📅 Work tomorrow' : '🎉 Free tomorrow'}
              </div>
            )}
          </div>
        </div>

        <div className="bedtime-settings">
          <h2>Bedtime Settings</h2>

          <div className="setting-group">
            <label>
              <div className="label-header">
                <span className="label-icon">💼</span>
                <span className="label-text">Work night bedtime</span>
              </div>
              <div className="label-description">
                When you have tasks tomorrow
              </div>
            </label>
            <input
              type="time"
              value={workNightBedtime}
              onChange={(e) => setWorkNightBedtime(e.target.value)}
              className="time-input"
            />
          </div>

          <div className="setting-group">
            <label>
              <div className="label-header">
                <span className="label-icon">🎮</span>
                <span className="label-text">Free night bedtime</span>
              </div>
              <div className="label-description">
                When you have no tasks tomorrow
              </div>
            </label>
            <input
              type="time"
              value={freeNightBedtime}
              onChange={(e) => setFreeNightBedtime(e.target.value)}
              className="time-input"
            />
          </div>
        </div>

        <div className="bedtime-preview">
          <div className="preview-label">Tonight's bedtime</div>
          <div className="preview-time">{getTargetBedtime()}</div>
          <div className="preview-reason">
            {hasWorkTomorrow ? 'You have work tomorrow' : 'No work tomorrow'}
          </div>
        </div>

        <div className="info-card">
          <h3>How it works</h3>
          <ul>
            <li>
              <span className="info-icon">🔗</span>
              <span>Connect your calendar to detect work days</span>
            </li>
            <li>
              <span className="info-icon">⏰</span>
              <span>Set different bedtimes for work and free nights</span>
            </li>
            <li>
              <span className="info-icon">🌊</span>
              <span>Won't interrupt when you're in flow state</span>
            </li>
            <li>
              <span className="info-icon">🔔</span>
              <span>Get gentle reminders when it's bedtime</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  )
}

export default App
