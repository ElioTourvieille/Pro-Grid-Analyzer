// API client for SmartPokerGrid backend

export interface PlayerProfile {
  abi: number;
  roi: number;
  style: string;
  hoursPerWeek: number;
  tournamentTypes: string[];
}

export interface TournamentData {
  name: string;
  buyIn: number;
  prizePool: number;
  players: number;
  startTime: string;
  type: string;
}

export interface AnalysisResult {
  tournamentId: string;
  tournamentName: string;
  buyIn: number;
  predictedRoi: number;
  confidence: number;
  reasoning: string;
  recommendation: 'high' | 'medium' | 'low';
}

export interface AnalysisRequest {
  playerProfile: PlayerProfile;
  tournamentData: TournamentData[];
}

export interface AnalysisResponse {
  results: AnalysisResult[];
  summary: {
    totalTournaments: number;
    recommendedTournaments: number;
    averagePredictedRoi: number;
  };
}

function mapTournamentDataToBackend(t: TournamentData) {
  return {
    "Nom": t.name,
    "Mise": t.buyIn,
    "Participants": t.players,
    "Date :": t.startTime,
    "Rake": 0,
    "Compétence": 70,
    "Compétence moyenne": 65,
    "Nb_jeux": 50,
    "Type": t.type,
    "Prix": t.prizePool,
  };
}

function mapProfileToBackend(p: PlayerProfile) {
  return {
    ABI: p.abi,
    ROI_total: p.roi,
    Style: p.style,
    heures: p.tournamentTypes, // Le backend attend 'heures' pour les types
  };
}

function mapAnalysisRequestToBackend(data: AnalysisRequest) {
  return {
    profil_joueur: mapProfileToBackend(data.playerProfile),
    tournois: data.tournamentData.map(mapTournamentDataToBackend),
  };
}

const API_BASE_URL = import.meta.env.PUBLIC_API_URL || 'http://localhost:8000';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  async analyzeTournaments(data: AnalysisRequest): Promise<AnalysisResponse> {
    const backendData = mapAnalysisRequestToBackend(data);
    const backendResponse = await this.request<any>('/api/predict', {
      method: 'POST',
      body: JSON.stringify(backendData),
    });
    return backendResponse;
  }

  async uploadTournaments(file: File): Promise<TournamentData[]> {
    const formData = new FormData();
    formData.append('file', file);

    return this.request<TournamentData[]>('/api/upload', {
      method: 'POST',
      body: formData,
      headers: {
        // Remove Content-Type for FormData
      },
    });
  }

  async analyzeFile(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);

    const backendResponse = await this.request<any>('/api/analyze', {
      method: 'POST',
      body: formData,
      headers: {
        // Remove Content-Type for FormData
      },
    });
    return backendResponse;
  }


}

export const apiClient = new ApiClient(); 