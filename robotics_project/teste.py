import React, { useState, useEffect } from 'react';

const TapatanGameSimulator = () => {
  // Game state
  const [board, setBoard] = useState(Array(9).fill(0)); // 0=empty, 1=player1, 2=player2
  const [currentPlayer, setCurrentPlayer] = useState(1);
  const [gamePhase, setGamePhase] = useState('placement'); // 'placement' or 'movement'
  const [piecesPlaced, setPiecesPlaced] = useState({ 1: 0, 2: 0 });
  const [selectedPiece, setSelectedPiece] = useState(null);
  const [winner, setWinner] = useState(null);
  const [gameLog, setGameLog] = useState([]);
  const [roboticSequence, setRoboticSequence] = useState([]);

  // Adjacency map for Tapatan (based on traditional board connections)
  const adjacencyMap = {
    0: [1, 3, 4],
    1: [0, 2, 3, 4, 5],
    2: [1, 4, 5],
    3: [0, 1, 4, 6, 7],
    4: [0, 1, 2, 3, 5, 6, 7, 8], // Center connects to all
    5: [1, 2, 4, 7, 8],
    6: [3, 4, 7],
    7: [3, 4, 5, 6, 8],
    8: [4, 5, 7]
  };

  // Winning patterns
  const winPatterns = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // Horizontal
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // Vertical
    [0, 4, 8], [2, 4, 6] // Diagonal
  ];

  // 3D coordinates for robotic arm (example positions)
  const roboticCoordinates = [
    [0.1, 0.1, 0.0],   // Position 0
    [0.1, 0.0, 0.0],   // Position 1
    [0.1, -0.1, 0.0],  // Position 2
    [0.0, 0.1, 0.0],   // Position 3
    [0.0, 0.0, 0.0],   // Position 4
    [0.0, -0.1, 0.0],  // Position 5
    [-0.1, 0.1, 0.0],  // Position 6
    [-0.1, 0.0, 0.0],  // Position 7
    [-0.1, -0.1, 0.0]  // Position 8
  ];

  // Check for winner
  const checkWinner = (boardState) => {
    for (const pattern of winPatterns) {
      const [a, b, c] = pattern;
      if (boardState[a] !== 0 && boardState[a] === boardState[b] && boardState[b] === boardState[c]) {
        return boardState[a];
      }
    }
    return null;
  };

  // Generate robotic sequence for placement
  const generatePlacementSequence = (position) => {
    const coord = roboticCoordinates[position];
    return [
      `Move above position ${position}: [${coord[0].toFixed(2)}, ${coord[1].toFixed(2)}, ${(coord[2] + 0.05).toFixed(2)}, 0, 0, 0]`,
      `Lower to place: [${coord[0].toFixed(2)}, ${coord[1].toFixed(2)}, ${(coord[2] + 0.02).toFixed(2)}, 0, 0, 0]`,
      `Lift after placing: [${coord[0].toFixed(2)}, ${coord[1].toFixed(2)}, ${(coord[2] + 0.05).toFixed(2)}, 0, 0, 0]`
    ];
  };

  // Generate robotic sequence for movement
  const generateMovementSequence = (from, to) => {
    const fromCoord = roboticCoordinates[from];
    const toCoord = roboticCoordinates[to];
    return [
      `Move above origin ${from}: [${fromCoord[0].toFixed(2)}, ${fromCoord[1].toFixed(2)}, ${(fromCoord[2] + 0.05).toFixed(2)}, 0, 0, 0]`,
      `Lower to pick: [${fromCoord[0].toFixed(2)}, ${fromCoord[1].toFixed(2)}, ${(fromCoord[2] + 0.02).toFixed(2)}, 0, 0, 0]`,
      `Lift piece: [${fromCoord[0].toFixed(2)}, ${fromCoord[1].toFixed(2)}, ${(fromCoord[2] + 0.05).toFixed(2)}, 0, 0, 0]`,
      `Move above destination ${to}: [${toCoord[0].toFixed(2)}, ${toCoord[1].toFixed(2)}, ${(toCoord[2] + 0.05).toFixed(2)}, 0, 0, 0]`,
      `Lower to place: [${toCoord[0].toFixed(2)}, ${toCoord[1].toFixed(2)}, ${(toCoord[2] + 0.02).toFixed(2)}, 0, 0, 0]`,
      `Lift after placing: [${toCoord[0].toFixed(2)}, ${toCoord[1].toFixed(2)}, ${(toCoord[2] + 0.05).toFixed(2)}, 0, 0, 0]`
    ];
  };

  // Handle cell click
  const handleCellClick = (position) => {
    if (winner) return;

    if (gamePhase === 'placement') {
      // Placement phase
      if (board[position] === 0) {
        const newBoard = [...board];
        newBoard[position] = currentPlayer;
        setBoard(newBoard);
        
        const newPiecesPlaced = { ...piecesPlaced };
        newPiecesPlaced[currentPlayer]++;
        setPiecesPlaced(newPiecesPlaced);

        // Generate robotic sequence
        const sequence = generatePlacementSequence(position);
        setRoboticSequence(sequence);

        // Add to game log
        setGameLog(prev => [...prev, `Player ${currentPlayer} places piece at position ${position}`]);

        // Check if placement phase is complete
        if (newPiecesPlaced[1] === 3 && newPiecesPlaced[2] === 3) {
          setGamePhase('movement');
          setGameLog(prev => [...prev, "All pieces placed! Movement phase begins."]);
        }

        // Check for winner
        const gameWinner = checkWinner(newBoard);
        if (gameWinner) {
          setWinner(gameWinner);
          setGameLog(prev => [...prev, `Game Over! Player ${gameWinner} wins!`]);
          return;
        }

        // Switch player
        setCurrentPlayer(currentPlayer === 1 ? 2 : 1);
      }
    } else if (gamePhase === 'movement') {
      // Movement phase
      if (selectedPiece === null) {
        // Select piece to move
        if (board[position] === currentPlayer) {
          setSelectedPiece(position);
          setGameLog(prev => [...prev, `Player ${currentPlayer} selects piece at position ${position}`]);
        }
      } else {
        // Move selected piece
        if (position === selectedPiece) {
          // Deselect
          setSelectedPiece(null);
          setGameLog(prev => [...prev, `Player ${currentPlayer} deselects piece`]);
        } else if (board[position] === 0 && adjacencyMap[selectedPiece].includes(position)) {
          // Valid move
          const newBoard = [...board];
          newBoard[selectedPiece] = 0;
          newBoard[position] = currentPlayer;
          setBoard(newBoard);

          // Generate robotic sequence
          const sequence = generateMovementSequence(selectedPiece, position);
          setRoboticSequence(sequence);

          setGameLog(prev => [...prev, `Player ${currentPlayer} moves from ${selectedPiece} to ${position}`]);
          setSelectedPiece(null);

          // Check for winner
          const gameWinner = checkWinner(newBoard);
          if (gameWinner) {
            setWinner(gameWinner);
            setGameLog(prev => [...prev, `Game Over! Player ${gameWinner} wins!`]);
            return;
          }

          // Switch player
          setCurrentPlayer(currentPlayer === 1 ? 2 : 1);
        } else {
          setSelectedPiece(null);
          setGameLog(prev => [...prev, "Invalid move! Piece deselected."]);
        }
      }
    }
  };

  // Reset game
  const resetGame = () => {
    setBoard(Array(9).fill(0));
    setCurrentPlayer(1);
    setGamePhase('placement');
    setPiecesPlaced({ 1: 0, 2: 0 });
    setSelectedPiece(null);
    setWinner(null);
    setGameLog(['Game started! Players alternate placing 3 pieces each.']);
    setRoboticSequence([]);
  };

  // Initialize game log
  useEffect(() => {
    if (gameLog.length === 0) {
      setGameLog(['Game started! Players alternate placing 3 pieces each.']);
    }
  }, []);

  // Get cell style based on state
  const getCellStyle = (position) => {
    let baseStyle = "w-16 h-16 border-2 border-gray-400 flex items-center justify-center text-2xl font-bold cursor-pointer transition-all duration-200 ";
    
    if (board[position] === 1) {
      baseStyle += "bg-blue-100 text-blue-600 ";
    } else if (board[position] === 2) {
      baseStyle += "bg-red-100 text-red-600 ";
    } else {
      baseStyle += "bg-gray-50 hover:bg-gray-100 ";
    }

    if (selectedPiece === position) {
      baseStyle += "ring-4 ring-yellow-400 ";
    }

    // Highlight valid moves in movement phase
    if (gamePhase === 'movement' && selectedPiece !== null && board[position] === 0 && adjacencyMap[selectedPiece].includes(position)) {
      baseStyle += "bg-green-100 hover:bg-green-200 ";
    }

    return baseStyle;
  };

  // Get piece symbol
  const getPieceSymbol = (position) => {
    if (board[position] === 1) return 'X';
    if (board[position] === 2) return 'O';
    return '';
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">Tapatan Game Simulator</h1>
        <p className="text-gray-600">Traditional Filipino Strategy Game - Robotic Implementation</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Game Board */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="text-center mb-4">
            <h2 className="text-2xl font-semibold text-gray-800">Game Board</h2>
            <div className="mt-2">
              <span className="text-lg">Phase: <span className="font-bold capitalize text-indigo-600">{gamePhase}</span></span>
              {!winner && (
                <span className="ml-4 text-lg">
                  Current Player: <span className={`font-bold ${currentPlayer === 1 ? 'text-blue-600' : 'text-red-600'}`}>
                    Player {currentPlayer} ({currentPlayer === 1 ? 'X' : 'O'})
                  </span>
                </span>
              )}
              {winner && (
                <div className="text-xl font-bold text-green-600 mt-2">
                  🎉 Player {winner} ({winner === 1 ? 'X' : 'O'}) Wins! 🎉
                </div>
              )}
            </div>
          </div>

          {/* Traditional Tapatan Board Layout */}
          <div className="flex justify-center mb-4">
            <div className="relative">
              {/* Board grid */}
              <div className="grid grid-cols-3 gap-1">
                {Array.from({ length: 9 }, (_, i) => (
                  <div
                    key={i}
                    className={getCellStyle(i)}
                    onClick={() => handleCellClick(i)}
                  >
                    {getPieceSymbol(i)}
                  </div>
                ))}
              </div>
              
              {/* Traditional Tapatan lines (SVG overlay) */}
              <svg 
                className="absolute inset-0 pointer-events-none" 
                viewBox="0 0 196 196"
                style={{ width: '196px', height: '196px' }}
              >
                {/* Diagonal lines */}
                <line x1="16" y1="16" x2="180" y2="180" stroke="#666" strokeWidth="1" opacity="0.3" />
                <line x1="180" y1="16" x2="16" y2="180" stroke="#666" strokeWidth="1" opacity="0.3" />
                {/* Center star lines */}
                <line x1="98" y1="16" x2="98" y2="180" stroke="#666" strokeWidth="1" opacity="0.3" />
                <line x1="16" y1="98" x2="180" y2="98" stroke="#666" strokeWidth="1" opacity="0.3" />
              </svg>
            </div>
          </div>

          {/* Game Status */}
          <div className="bg-gray-50 rounded-lg p-4 mb-4">
            <div className="grid grid-cols-2 gap-4 text-center">
              <div>
                <div className="text-blue-600 font-semibold">Player 1 (X)</div>
                <div>Pieces: {piecesPlaced[1]}/3</div>
              </div>
              <div>
                <div className="text-red-600 font-semibold">Player 2 (O)</div>
                <div>Pieces: {piecesPlaced[2]}/3</div>
              </div>
            </div>
          </div>

          <div className="text-center">
            <button
              onClick={resetGame}
              className="bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700 transition-colors"
            >
              Reset Game
            </button>
          </div>
        </div>

        {/* Game Information */}
        <div className="space-y-6">
          {/* Robotic Sequence */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">🤖 Robotic Arm Sequence</h3>
            <div className="bg-gray-50 rounded-lg p-4 max-h-48 overflow-y-auto">
              {roboticSequence.length > 0 ? (
                <div className="space-y-2">
                  {roboticSequence.map((step, index) => (
                    <div key={index} className="text-sm font-mono text-gray-700 bg-white p-2 rounded">
                      {index + 1}. {step}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-gray-500 text-center">Make a move to see robotic sequence</div>
              )}
            </div>
          </div>

          {/* Game Log */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">📝 Game Log</h3>
            <div className="bg-gray-50 rounded-lg p-4 max-h-64 overflow-y-auto">
              {gameLog.map((entry, index) => (
                <div key={index} className="text-sm text-gray-700 mb-1">
                  <span className="text-gray-500">{index + 1}.</span> {entry}
                </div>
              ))}
            </div>
          </div>

          {/* Game Rules */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">📋 Tapatan Rules</h3>
            <div className="text-sm text-gray-700 space-y-2">
              <div><strong>Phase 1 - Placement:</strong> Each player places 3 pieces alternately</div>
              <div><strong>Phase 2 - Movement:</strong> Move pieces to adjacent empty positions</div>
              <div><strong>Winning:</strong> Get 3 pieces in a row (horizontal, vertical, or diagonal)</div>
              <div><strong>Movement:</strong> Pieces can only move along the board's connection lines</div>
              <div><strong>Note:</strong> No piece capture - pure positional strategy!</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TapatanGameSimulator;