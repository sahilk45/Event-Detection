import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  LinearProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Mic,
  PlayArrow,
  Stop,
  Warning,
  CheckCircle,
  Schedule,
} from '@mui/icons-material';
import RecordRTC from 'recordrtc';

const LiveRecording = ({ onPrediction }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [hasPermission, setHasPermission] = useState(false);
  const [currentChunk, setCurrentChunk] = useState(0);
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState(null);
  const [audioLevel, setAudioLevel] = useState(0);
  const [continuousMode, setContinuousMode] = useState(false);

  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const chunkIntervalRef = useRef(null);
  const animationFrameRef = useRef(null);

  const CHUNK_DURATION = 5000; // 5 seconds per chunk

 useEffect(() => {
  return () => {
    // Cleanup on unmount - direct cleanup without calling stopRecording
    if (recorderRef.current && recorderRef.current.getState() === 'recording') {
      recorderRef.current.stopRecording();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    if (chunkIntervalRef.current) {
      clearInterval(chunkIntervalRef.current);
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
  };
}, []); // Empty dependency array is fine with this approach


  const requestMicrophonePermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 22050,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        } 
      });
      
      streamRef.current = stream;
      setHasPermission(true);
      setError(null);
      
      // Setup audio level monitoring
      setupAudioLevelMonitoring(stream);
      
      return stream;
    } catch (err) {
      setError('Microphone permission denied. Please allow microphone access and try again.');
      setHasPermission(false);
      return null;
    }
  };

  const setupAudioLevelMonitoring = (stream) => {
    audioContextRef.current = new AudioContext();
    analyserRef.current = audioContextRef.current.createAnalyser();
    const source = audioContextRef.current.createMediaStreamSource(stream);
    source.connect(analyserRef.current);
    
    analyserRef.current.fftSize = 256;
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const updateAudioLevel = () => {
      analyserRef.current.getByteFrequencyData(dataArray);
      const average = dataArray.reduce((sum, value) => sum + value, 0) / bufferLength;
      setAudioLevel(Math.min(100, (average / 255) * 100));
      
      if (isRecording) {
        animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
      }
    };
    
    updateAudioLevel();
  };

  const startRecording = async () => {
    let stream = streamRef.current;
    
    if (!stream) {
      stream = await requestMicrophonePermission();
      if (!stream) return;
    }

    try {
      // Initialize RecordRTC
      recorderRef.current = new RecordRTC(stream, {
        type: 'audio',
        mimeType: 'audio/wav',
        sampleRate: 22050,
        numberOfAudioChannels: 1,
        timeSlice: CHUNK_DURATION,
        ondataavailable: handleChunkAvailable,
      });

      recorderRef.current.startRecording();
      setIsRecording(true);
      setCurrentChunk(0);
      setPredictions([]);
      setError(null);

      // Start chunk processing interval
      if (continuousMode) {
        chunkIntervalRef.current = setInterval(() => {
          processCurrentChunk();
        }, CHUNK_DURATION);
      }

    } catch (err) {
      setError('Failed to start recording: ' + err.message);
    }
  };

  const stopRecording = () => {
    if (recorderRef.current && isRecording) {
      recorderRef.current.stopRecording(() => {
        // Process final chunk if not in continuous mode
        if (!continuousMode) {
          const blob = recorderRef.current.getBlob();
          processAudioChunk(blob, currentChunk);
        }
      });
    }

    setIsRecording(false);
    
    if (chunkIntervalRef.current) {
      clearInterval(chunkIntervalRef.current);
    }
    
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
  };

  const processCurrentChunk = () => {
    if (recorderRef.current && isRecording) {
      // Stop current recording to get the chunk
      recorderRef.current.stopRecording(() => {
        const blob = recorderRef.current.getBlob();
        processAudioChunk(blob, currentChunk);
        
        // Start a new recording for the next chunk
        setTimeout(() => {
          if (isRecording && streamRef.current) {
            recorderRef.current = new RecordRTC(streamRef.current, {
              type: 'audio',
              mimeType: 'audio/wav',
              sampleRate: 22050,
              numberOfAudioChannels: 1,
            });
            recorderRef.current.startRecording();
            setCurrentChunk(prev => prev + 1);
          }
        }, 100);
      });
    }
  };

  const handleChunkAvailable = (blob) => {
    if (continuousMode) {
      processAudioChunk(blob, currentChunk);
      setCurrentChunk(prev => prev + 1);
    }
  };

const processAudioChunk = async (audioBlob, chunkIndex) => {
  try {
    // Validate blob size before sending
    if (audioBlob.size < 500) {
      console.log(`⚠️ Skipping tiny audio chunk ${chunkIndex} (${audioBlob.size} bytes)`);
      return;
    }

    const formData = new FormData();
    const audioFile = new File([audioBlob], `chunk_${chunkIndex}.wav`, {
      type: 'audio/wav'
    });
    formData.append('file', audioFile);

    const response = await fetch('http://localhost:8000/api/v1/predict', {
      method: 'POST',
      body: formData,
    });

    // Handle both successful and error responses
    const result = await response.json();
    
    // Skip error responses gracefully
    if (result.predicted_category === 'error' || result.predicted_category === 'silence') {
      console.log(`⚠️ Skipping chunk ${chunkIndex}: ${result.predicted_category}`);
      return;
    }
    
    // Only process high-confidence predictions
    if (result.confidence > 0.70) {
      const predictionWithTimestamp = {
        ...result,
        chunkIndex,
        timestamp: new Date().toISOString(),
        timeOffset: chunkIndex * (CHUNK_DURATION / 1000),
      };

      setPredictions(prev => [...prev, predictionWithTimestamp]);
      
      if (onPrediction) {
        onPrediction(predictionWithTimestamp);
      }

      // Log high-confidence events
      if (result.alert_info?.should_alert) {
        const eventLog = JSON.parse(localStorage.getItem('eventLog') || '[]');
        eventLog.push({
          event: result.alert_info.event_name,
          confidence: result.alert_info.confidence,
          priority: result.alert_info.priority,
          timestamp: predictionWithTimestamp.timestamp,
          filename: `Live Recording - Chunk ${chunkIndex}`,
          source: 'live_recording'
        });
        
        if (eventLog.length > 50) {
          eventLog.splice(0, eventLog.length - 50);
        }
        
        localStorage.setItem('eventLog', JSON.stringify(eventLog));
        window.dispatchEvent(new Event('storage'));
      }
    }

  } catch (err) {
    console.error(`Error processing chunk ${chunkIndex}:`, err);
    // Don't show error to user for individual chunk failures
    // Live recording should continue despite occasional chunk errors
  }
};



  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getPriorityColor = (priority) => {
    const colors = {
      'critical': 'error',
      'high': 'warning',
      'medium': 'info',
      'low': 'success',
    };
    return colors[priority] || 'default';
  };

  return (
    <Card>
      <CardContent sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
          <Mic color="primary" sx={{ fontSize: 32 }} />
          <Typography variant="h5" fontWeight={700}>
            Live Audio Detection
          </Typography>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {!hasPermission ? (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
              Allow microphone access to start live audio detection
            </Typography>
            <Button
              variant="contained"
              startIcon={<Mic />}
              onClick={requestMicrophonePermission}
              size="large"
            >
              Grant Microphone Permission
            </Button>
          </Box>
        ) : (
          <Box>
            {/* Controls */}
            <Box sx={{ display: 'flex', gap: 2, mb: 3, alignItems: 'center' }}>
              <Button
                variant={isRecording ? "outlined" : "contained"}
                color={isRecording ? "error" : "primary"}
                startIcon={isRecording ? <Stop /> : <PlayArrow />}
                onClick={isRecording ? stopRecording : startRecording}
                size="large"
              >
                {isRecording ? 'Stop Recording' : 'Start Recording'}
              </Button>

              <FormControlLabel
                control={
                  <Switch
                    checked={continuousMode}
                    onChange={(e) => setContinuousMode(e.target.checked)}
                    disabled={isRecording}
                  />
                }
                label="Continuous Mode"
              />
            </Box>

            {/* Audio Level Indicator */}
            {isRecording && (
              <Box sx={{ mb: 3 }}>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  Audio Level: {audioLevel.toFixed(0)}%
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={audioLevel}
                  sx={{ 
                    height: 8, 
                    borderRadius: 4,
                    backgroundColor: 'grey.700'
                  }}
                />
              </Box>
            )}

            {/* Recording Status */}
            {isRecording && (
              <Box sx={{ mb: 3 }}>
                <Alert 
                  severity="info" 
                  icon={<Mic />}
                  sx={{
                    '& .MuiAlert-message': { 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: 1 
                    }
                  }}
                >
                  <Typography variant="body1" fontWeight={600}>
                    Recording in progress... Chunk: {currentChunk + 1}
                  </Typography>
                  <Chip 
                    label="LIVE"
                    color="error"
                    size="small"
                    sx={{ 
                      animation: 'pulse 1.5s infinite',
                      '@keyframes pulse': {
                        '0%, 100%': { opacity: 1 },
                        '50%': { opacity: 0.5 }
                      }
                    }}
                  />
                </Alert>
              </Box>
            )}

            {/* High-Confidence Predictions */}
            {predictions.length > 0 && (
              <Box>
                <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
                  High-Confidence Detections ({predictions.length})
                </Typography>
                
                <List sx={{ maxHeight: 300, overflow: 'auto' }}>
                  {predictions.map((prediction, index) => (
                    <Box key={`${prediction.chunkIndex}-${index}`}>
                      <ListItem 
                        sx={{ 
                          bgcolor: prediction.alert_info?.should_alert ? 'warning.main' : 'background.paper',
                          borderRadius: 1,
                          mb: 1,
                          color: prediction.alert_info?.should_alert ? 'warning.contrastText' : 'text.primary'
                        }}
                      >
                        <ListItemIcon>
                          {prediction.alert_info?.should_alert ? (
                            <Warning color="inherit" />
                          ) : (
                            <CheckCircle color="success" />
                          )}
                        </ListItemIcon>
                        
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="body1" fontWeight={600}>
                                {prediction.predicted_category}
                                {prediction.predicted_subcategory && ` → ${prediction.predicted_subcategory}`}
                              </Typography>
                              <Chip 
                                label={`${(prediction.confidence * 100).toFixed(1)}%`}
                                size="small"
                                color={prediction.alert_info?.should_alert ? 'error' : 'success'}
                                sx={{ fontWeight: 600 }}
                              />
                              {prediction.alert_info?.should_alert && (
                                <Chip 
                                  label={prediction.alert_info.priority.toUpperCase()}
                                  size="small"
                                  color={getPriorityColor(prediction.alert_info.priority)}
                                />
                              )}
                            </Box>
                          }
                          secondary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                              <Schedule sx={{ fontSize: 14 }} />
                              <Typography variant="caption">
                                {formatTime(prediction.timeOffset)} • {new Date(prediction.timestamp).toLocaleTimeString()}
                              </Typography>
                            </Box>
                          }
                        />
                      </ListItem>
                      {index < predictions.length - 1 && <Divider />}
                    </Box>
                  ))}
                </List>
              </Box>
            )}

            {/* Instructions */}
            <Box sx={{ mt: 3, p: 2, bgcolor: 'background.paper', borderRadius: 2 }}>
              <Typography variant="body2" color="text.secondary">
                <strong>Instructions:</strong><br/>
                • <strong>Single Recording:</strong> Turn off continuous mode, click start/stop manually<br/>
                • <strong>Continuous Mode:</strong> Automatically processes 5-second chunks while recording<br/>
                • Only predictions with &gt; 70% confidence will be displayed<br/>
                • High-priority alerts will trigger notifications and be logged
              </Typography>
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default LiveRecording;
