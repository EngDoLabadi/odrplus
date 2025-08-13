import {
  type Message,
  convertToCoreMessages,
  createDataStreamResponse,
  generateObject,
  generateText,
  streamObject,
  streamText,
} from 'ai';
import { z } from 'zod';

import { auth, signIn } from '@/app/(auth)/auth';
import { customModel } from '@/lib/ai';
import { models, reasoningModels } from '@/lib/ai/models';
import { rateLimiter } from '@/lib/rate-limit';
import {
  codePrompt,
  systemPrompt,
  updateDocumentPrompt,
} from '@/lib/ai/prompts';
import {
  deleteChatById,
  getChatById,
  getDocumentById,
  getUser,
  saveChat,
  saveDocument,
  saveMessages,
  saveSuggestions,
} from '@/lib/db/queries';
import type { Suggestion } from '@/lib/db/schema';
import {
  generateUUID,
  getMostRecentUserMessage,
  sanitizeResponseMessages,
} from '@/lib/utils';

import { generateTitleFromUserMessage } from '../../actions';
import FirecrawlApp from '@mendable/firecrawl-js';

type AllowedTools = 'deepResearch' | 'search' | 'extract' | 'scrape';

const firecrawlTools: AllowedTools[] = ['search', 'extract', 'scrape'];
const allTools: AllowedTools[] = [...firecrawlTools, 'deepResearch'];

const app = new FirecrawlApp({
  apiKey: process.env.FIRECRAWL_API_KEY || '',
});

// Common types and interfaces
interface ResearchState {
  findings: Array<{ text: string; source: string }>;
  summaries: string[];
  nextSearchTopic: string;
  urlToSearch?: string;
  currentDepth: number;
  failedAttempts: number;
  maxFailedAttempts: number;
  processedUrls?: Set<string>;
  subquestions?: string[];
  answeredSubquestions?: string[];
  subAnswers?: Array<{ query: string; answer: string }>;
  completedSteps?: number;
  totalExpectedSteps?: number;
  urlFrequencyMap?: Map<string, { url: string; frequency: number; title?: string }>;
}

interface AnalysisResult {
  summary: string;
  hasAnswer: boolean;
  confidence: string;
  gaps: string[];
  shouldContinue: boolean;
  nextSearchTopic?: string;
  urlToSearch?: string;
  subquestions?: string[];
  subAnswer?: string;
  lastQuery?: string;
}

const ensureValidBrowseCompFormat = (text: string, query: string): string => {
  if (!text) {
    return `Explanation: The research could not find a definitive answer to: "${query}".
Exact Answer: Unknown
Confidence: 10%`;
  }

  const lines = text.trim().split('\n').map(line => line.trim());
  if (lines.length === 3 && 
      lines[0].toLowerCase().startsWith('explanation:') &&
      lines[1].toLowerCase().startsWith('exact answer:') &&
      /^confidence:\s*(100|[1-9]?[0-9])%$/i.test(lines[2])) {
    return text.trim();
  }

  let explanation = '', exactAnswer = '', confidence = '';
  
  const explanationMatch = text.match(/explanation:\s*(.*?)(?=exact answer:|confidence:|$)/is);
  if (explanationMatch) explanation = explanationMatch[1].trim();
  
  const exactAnswerMatch = text.match(/exact answer:\s*(.*?)(?=explanation:|confidence:|$)/is);
  if (exactAnswerMatch) exactAnswer = exactAnswerMatch[1].trim();
  
  const confidenceMatch = text.match(/confidence:\s*(\d{1,3}%)/i);
  if (confidenceMatch) confidence = confidenceMatch[1].trim();

  if (explanation || exactAnswer || confidence) {
    explanation = explanation || 'The research could not find a definitive answer.';
    exactAnswer = exactAnswer || 'Unknown';
    confidence = confidence || '30%';
    
    if (!explanation.toLowerCase().startsWith('explanation:')) {
      explanation = 'Explanation: ' + explanation;
    }
    if (!exactAnswer.toLowerCase().startsWith('exact answer:')) {
      exactAnswer = 'Exact Answer: ' + exactAnswer;
    }
    if (!confidence.toLowerCase().startsWith('confidence:')) {
      confidence = 'Confidence: ' + confidence;
    }
    
    return `${explanation}\n${exactAnswer}\n${confidence}`;
  }
  
  return `Explanation: The research could not find a definitive answer to: "${query}".
Exact Answer: Unknown
Confidence: 10%`;
};

const parseAnalysisResponse = (responseText: string, findings: any[], timeRemainingMinutes: number, question: string): AnalysisResult => {
  // Method 1: Try direct JSON parsing
  try {
    const parsed = JSON.parse(responseText);
    if (parsed.analysis) {
      return parsed.analysis;
    }
    if (parsed.summary !== undefined) {
      return parsed;
    }
    if (parsed.subquestions && Array.isArray(parsed.subquestions)) {
      parsed.subquestions = parsed.subquestions.filter(q => typeof q === 'string');
    }
    return parsed;
  } catch (error) {
    console.log('Direct JSON parsing failed:', error.message);
  }

  // Method 2: Extract from markdown code blocks
  const markdownPatterns = [
    /```(?:json)?\s*({[\s\S]*?})\s*```/g,
    /```\s*({[\s\S]*?})\s*```/g,
  ];

  for (const pattern of markdownPatterns) {
    const matches = [...responseText.matchAll(pattern)];
    for (const match of matches) {
      try {
        const parsed = JSON.parse(match[1]);
        if (parsed.analysis) {
          return parsed.analysis;
        }
        if (parsed.summary !== undefined) {
          return parsed;
        }
      } catch (error) {
        continue;
      }
    }
  }

  // Method 3: Find any JSON object in the text
  const jsonMatches = responseText.match(/{[^{}]*(?:{[^{}]*}[^{}]*)*}/g);
  if (jsonMatches) {
    for (const jsonMatch of jsonMatches) {
      try {
        const parsed = JSON.parse(jsonMatch);
        if (parsed.analysis) {
          return parsed.analysis;
        }
        if (parsed.summary !== undefined || parsed.shouldContinue !== undefined) {
          return parsed;
        }
      } catch (error) {
        continue;
      }
    }
  }
  return parseAnalysisFromText(responseText, findings, timeRemainingMinutes, question);
};

const generateFallbackQuery = (question: string): string => {
      // Extract specific searchable facts without being domain-specific
      const years = question.match(/\b(19|20)\d{2}\b/g);
      const percentages = question.match(/\b\d+\.?\d*%/g);
      const numbers = question.match(/\b\d+\s+\w+/gi); // Any number with context
      
      const entities = [];
      
      // Extract quoted phrases (often the most important clues)
      const quotes = question.match(/"([^"]+)"/g);
      if (quotes) entities.push(...quotes.map(q => q.replace(/"/g, '')));
      
      // Extract capitalized entities (but filter out format instructions)
      const caps = question.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b/g);
      if (caps) {
        entities.push(...caps.filter(c => 
          !c.toLowerCase().includes('explanation') && 
          !c.toLowerCase().includes('answer') &&
          !c.toLowerCase().includes('confidence') &&
          !c.toLowerCase().includes('exact')
        ));
      }
      
      // Add specific factual details
      if (years) entities.push(...years);
      if (percentages) entities.push(...percentages);
      if (numbers) entities.push(...numbers.slice(0, 2)); // First 2 numbers with context
      
      // If we have good entities, use them
      if (entities.length > 0) {
        return entities.slice(0, 5).join(' ');
      }
      
      // Final fallback: extract meaningful words (but avoid format instructions)
      const words = question
        .toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(w => 
          w.length > 4 && 
          !w.includes('explanation') && 
          !w.includes('answer') && 
          !w.includes('confidence') &&
          !w.includes('exact') &&
          !w.includes('format')
        )
        .slice(0, 4);
        
      return words.join(' ') || 'search query';
    };

const parseAnalysisFromText = (text: string, findings: any[], timeRemainingMinutes: number, question: string): AnalysisResult => {
  const lowerText = text.toLowerCase();
  
  const shouldContinue = 
    findings.length < 3 || 
    timeRemainingMinutes > 1.5 ||
    lowerText.includes('continue') ||
    lowerText.includes('more search') ||
    lowerText.includes('insufficient');

  let confidence = 'low';
  if (lowerText.includes('high confidence') || lowerText.includes('confident')) {
    confidence = 'high';
  } else if (lowerText.includes('medium') || lowerText.includes('moderate')) {
    confidence = 'medium';
  }

  const hasAnswer = 
    lowerText.includes('found') ||
    lowerText.includes('answer') ||
    lowerText.includes('identified') ||
    confidence === 'high';

  return {
    summary: text.substring(0, 200),
    hasAnswer,
    confidence,
    gaps: ['More information needed'],
    shouldContinue,
    nextSearchTopic: generateFallbackQuery(question),
  };
};

const createFallbackAnalysis = (findings: any[], timeRemainingMinutes: number, question: string): AnalysisResult => {
  return {
    summary: "Analysis function encountered an error",
    hasAnswer: false,
    confidence: "low",
    gaps: ["Analysis system error"],
    shouldContinue: findings.length < 5 && timeRemainingMinutes > 1,
    nextSearchTopic: generateFallbackQuery(question)
  };
};

const extractKeyTerms = (question: string): string => {
  const stopwords = new Set([
    'the', 'a', 'an', 'and', 'or', 'but',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'is', 'was', 'are', 'were', 'be', 'been', 'being',
    'has', 'have', 'had', 'can', 'could', 'would', 'should',
    'what', 'who', 'where', 'when', 'which', 'how', 'why',
    'your', 'response', 'explanation', 'answer', 'confidence',
    'additionally', 'also', 'both', 'either', 'neither'
  ]);

  const keywords = new Set<string>();

  // Extract quoted phrases
  for (const match of question.matchAll(/"([^"]+)"/g)) {
    if (match[1].length > 2) keywords.add(match[1].trim());
  }

  // Extract capitalized multi-word names/entities
  for (const match of question.matchAll(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b/g)) {
    const phrase = match[0].trim();
    if (phrase.length > 2 && !stopwords.has(phrase.toLowerCase())) {
      keywords.add(phrase);
    }
  }

  // Extract numbers, years, percentages
  (question.match(/\b(19[5-9]\d|20[0-4]\d)\b/g) || []).forEach(yr => keywords.add(yr));
  (question.match(/\b\d+(?:\.\d+)?%/g) || []).forEach(pct => keywords.add(pct));

  // Extract meaningful words
  question
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(w =>
      w.length > 4 &&
      !stopwords.has(w) &&
      !/^\d+$/.test(w) &&
      !w.includes('explanation') &&
      !w.includes('answer') &&
      !w.includes('confidence')
    )
    .slice(0, 8)
    .forEach(w => keywords.add(w));

  return Array.from(keywords)
    .sort((a, b) => b.length - a.length)
    .slice(0, 8)
    .join(' ');
};

const filterUrls = (urls: string[]): string[] => {
  const blockedDomains = [
    'reddit.com', 'brainly.com', 'youtube.com', 'youtu.be',
    'facebook.com', 'twitter.com', 'x.com', 'tiktok.com', 'instagram.com'
  ];

  return urls.filter(url => {
    try {
      const domain = new URL(url).hostname;
      if (url.match(/\.(pdf|doc|docx)$/i)) return false;
      return !blockedDomains.some(blocked => domain.includes(blocked));
    } catch {
      return false;
    }
  });
};

const isGeneric = (query: string): boolean => {
  return /^(what|when|where|who|how|name|info|event)$/i.test(query.trim()) ||
         query.split(' ').length < 3 ||
         query.match(/^\d+$/);
};
const searchWithRetry = async (query, retries = 3) => {
  for (let i = 0; i <= retries; i++) {
    try {
      const result = await app.search(query);
      if (result.success) return result;
      if (i < retries) {
        console.log(`Search attempt ${i + 1} failed, retrying in ${2000 * (i + 1)}ms...`);
        await new Promise(resolve => setTimeout(resolve, 2000 * (i + 1)));
      }
    } catch (error) {
      if (i === retries) {
        console.error(`All ${retries + 1} search attempts failed for query: "${query}"`);
        throw error;
      }
      console.log(`Search retry ${i + 1}/${retries + 1} after error: ${error.message}`);
      await new Promise(resolve => setTimeout(resolve, 2000 * (i + 1))); 
    }
  }
  
  throw new Error(`Search failed after ${retries + 1} attempts`);
}

const createAnalyzeAndPlan = (reasoningModel: any, startTime: number, timeLimit: number) => {
  return async (findings: any[], lastQuery: string, question: string, subAnswers: any[] = []): Promise<AnalysisResult> => {
    const timeElapsed = Date.now() - startTime;
    const timeRemaining = timeLimit - timeElapsed;
    const timeRemainingMinutes = Math.round((timeRemaining / 1000 / 60) * 10) / 10;

    try {
      const result = await generateText({
        model: customModel(reasoningModel.apiIdentifier, true),
        prompt: `You are analyzing research findings for this BrowseComp identification question: "${question}"

Current findings (${findings.length} sources):
${findings.map((f) => `[Source: ${f.source}]: ${f.text}`).join('\n')}

${subAnswers.length > 0 ? `Previous subquestion answers:
${subAnswers.map(a => `Q: ${a.query}\nA: ${a.answer}`).join('\n\n')}` : ''}

ANALYSIS INSTRUCTIONS:
1. Determine if we can identify the specific individual/entity described
2. Check if we have enough constraint matches to answer the original question
3. If you already found a candidate answer build subsequent subquestions around VERIFYING that candidate
4. Generate NEW subquestions that:
   - BUILD ON existing findings - if you found a specific name/entity, include it in new subquestions
   - Preserve the most identifying constraints from the original question
   - VERIFY the candidate found rather than starting over with generic searches
   - Combine multiple constraint types for precise identification
   - Progress toward answering the specific question asked
   - Are NOT generic - must include specific details that uniquely identify the target

CRITICAL: 
- Subquestions must NOT be generic
- If you found a candidate answer, BUILD ON IT in subsequent subquestions
- Include specific details from the original question that uniquely identify the target
- VERIFY existing findings rather than starting over

CONFIDENCE ASSESSMENT:
- HIGH: Multiple sources confirm same individual/entity with exact constraint matches
- MEDIUM: Good constraint matches but need verification of specific detail asked
- LOW: Found relevant information but constraints don't uniquely identify target

ANSWER STRATEGY:
- If you found a specific candidate answer in earlier findings, note it as a potential answer
- Continue searching to see if you can find better verification or alternative candidates  
- If no better answers emerge after additional searches, use the best candidate found so far
- Don't discard good candidates just because all constraints aren't perfectly verified

Respond with ONLY a JSON object (NO backticks, NO markdown):

{
  "summary": "brief summary focusing on constraint matching progress",
  "hasAnswer": false,
  "confidence": "low|medium|high",
  "gaps": ["specific missing constraints or verification needs"],
  "shouldContinue": true,
  "subquestions": ["constraint-preserving subquestion 1", "constraint-preserving subquestion 2"],
  "subAnswer": "answer to the subquestion just attempted (if determinable from findings)",
  "lastQuery": "${lastQuery}",
  "nextSearchTopic": "constraint-aware search terms for missing piece",
  "strategy": "how to use constraints to narrow down further"
}

Time remaining: ${timeRemainingMinutes} minutes`
      });

      return parseAnalysisResponse(result.text, findings, timeRemainingMinutes, question);
    } catch (error) {
      console.error('Analysis error:', error);
      return createFallbackAnalysis(findings, timeRemainingMinutes, question);
    }
  };
};

const extractWithRetry = async (url: string, prompt: string, retries = 2) => {
  for (let i = 0; i <= retries; i++) {
    try {
      const result = await Promise.race([
        app.extract([url], { prompt }),
        new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), 35000)),
      ]);

      if (result.success) {
        if (Array.isArray(result.data)) {
          return result.data.map((item) => {
            const text = typeof item.data === 'string' ? item.data : JSON.stringify(item.data);
            return { text, source: url };
          });
        }
        const text = typeof result.data === 'string' ? result.data : JSON.stringify(result.data);
        return [{ text, source: url }];
      }

      if (!result.success || !result.data || JSON.stringify(result.data).includes(`"names":[]`)) {
        console.log('Extract returned empty data, trying scrape fallback...');
        
        try {
          const scrapeResult = await app.scrapeUrl(url);
          if (scrapeResult.success && scrapeResult.markdown) {
            return [{
              text: scrapeResult.markdown.substring(0, 2000),
              source: url
            }];
          }
        } catch (scrapeError) {
          console.log('Scrape fallback also failed');
        }
      }
    } catch (err) {
      console.warn(`Retry ${i} failed for ${url}: ${err.message}`);
      await new Promise((res) => setTimeout(res, 1000 * 2 ** i));
    }
  }
  return [];
};

const extractFromUrls = async (urls: string[], extractionPrompt: string) => {
  const filteredUrls = filterUrls(urls);
  console.log(`Filtered ${urls.length} URLs to ${filteredUrls.length} safe URLs`);

  const results = [];
  
  for (let i = 0; i < filteredUrls.length; i++) {
    const url = filteredUrls[i];
    
    try {
      console.log(`Extracting from: ${url} (${i + 1}/${filteredUrls.length})`);
      
      if (i > 0) {
        console.log('Waiting 2 seconds...');
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
      
      // Use the constraint-focused extraction prompt
      const extractResults = await extractWithRetry(url, extractionPrompt);
      
      if (extractResults.length > 0) {
        results.push(...extractResults);
        console.log(`Successfully extracted from ${url}`);
      } else {
        console.log(`Extraction failed for ${url}`);
      }
      
    } catch (error) {
      console.warn(`Processing failed for ${url}: ${error.message}`);
    }
  }

  return results;
};

// Session management
const ensureAuthenticatedSession = async () => {
  let session = await auth();

  if (!session?.user) {
    try {
      const result = await signIn('credentials', { redirect: false });
      if (result?.error) {
        console.error('Failed to create anonymous session:', result.error);
        throw new Error('Failed to create anonymous session');
      }

      let retries = 3;
      while (retries > 0) {
        session = await auth();
        
        if (session?.user?.id) {
          const users = await getUser(session.user.email as string);
          if (users.length > 0) {
            break;
          }
        }
        
        await new Promise(resolve => setTimeout(resolve, 1000));
        retries--;
      }

      if (!session?.user) {
        console.error('Failed to get session after creation');
        throw new Error('Failed to create session');
      }
    } catch (error) {
      console.error('Error creating anonymous session:', error);
      throw error;
    }
  }

  if (!session?.user?.id) {
    throw new Error('Failed to create session');
  }

  // Verify user exists in database
  try {
    const users = await getUser(session.user.email as string);
    if (users.length === 0) {
      console.error('User not found in database:', session.user);
      throw new Error('User not found');
    }
  } catch (error) {
    console.error('Error verifying user:', error);
    throw new Error('Failed to verify user');
  }

  return session;
};

// BrowseComp handler
const handleBrowseCompRequest = async (messages: any[], modelId: string, reasoningModelId: string) => {
  console.log('BrowseComp request detected');
  const lastUserMessage = messages[messages.length - 1]?.content || '';
  
  if (!lastUserMessage) {
    return new Response(JSON.stringify({ 
      content: ensureValidBrowseCompFormat('', 'No message provided') 
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  try {
    const model = models.find((model) => model.id === modelId) || models[0];
    const reasoningModel = reasoningModels.find((model) => model.id === reasoningModelId) || reasoningModels[0];
    
    console.log('Executing deepResearch for BrowseComp question:', lastUserMessage);
    
    const startTime = Date.now();
    const timeLimit = 3.5 * 60 * 1000; // 3.5 minutes for BrowseComp
    const maxDepth = 6;

    const researchState: ResearchState = {
      findings: [],
      summaries: [],
      nextSearchTopic: '',
      currentDepth: 0,
      failedAttempts: 0,
      maxFailedAttempts: 3,
      processedUrls: new Set(),
      subquestions: [],
      answeredSubquestions: [],
      subAnswers: [],
      urlFrequencyMap: new Map(),
    };
    const analyzeAndPlan = createAnalyzeAndPlan(reasoningModel, startTime, timeLimit);

    const extractConstraintsWithLLM = async (question: string, reasoningModel: any): Promise<string[]> => {
      try {
        const result = await generateText({
          model: customModel(reasoningModel.apiIdentifier, true),
          prompt: `Extract the key identifying constraints from this BrowseComp question: "${question}"

    INSTRUCTIONS:
    1. Identify the MOST SPECIFIC details that uniquely identify the target
    2. Focus on constraints that narrow down the possibilities
    3. Include exact numbers, dates, quotes, names, and unique descriptors
    4. Ignore generic terms and formatting instructions
    5. Prioritize constraints that would be hard to guess or common

    Extract constraints as a JSON array of strings:`,
          maxTokens: 300,
        });

        // Parse the LLM response
        let constraints = [];
        try {
          // Try direct JSON parsing
          constraints = JSON.parse(result.text);
        } catch {
          // Try extracting from brackets
          const arrayMatch = result.text.match(/\[(.*?)\]/s);
          if (arrayMatch) {
            try {
              constraints = JSON.parse(arrayMatch[0]);
            } catch {
              // Extract quoted strings as fallback
              const quotes = result.text.match(/"([^"]+)"/g);
              if (quotes) {
                constraints = quotes.map(q => q.replace(/"/g, ''));
              }
            }
          }
        }

        // Filter and validate
        constraints = constraints
          .filter(c => typeof c === 'string' && c.length > 2 && c.length < 50)
          .filter(c => !c.toLowerCase().includes('explanation'))
          .filter(c => !c.toLowerCase().includes('confidence'))
          .slice(0, 8); 

        console.log(`LLM extracted ${constraints.length} constraints:`, constraints);
        return constraints;

      } catch (error) {
        console.warn('LLM constraint extraction failed, using fallback:', error.message);
        // Fallback to simple keyword extraction
        return extractKeyTerms(question).split(' ').slice(0, 5);
      }
    };

    const getUrlsToProcess = (urlFrequencyMap: Map<string, any>, processedUrls: Set<string>) => {
      const sortedUrls = Array.from(urlFrequencyMap.values())
        .sort((a, b) => b.frequency - a.frequency)
        .map(item => item.url)
        .filter(url => !processedUrls.has(url));
    
      return sortedUrls.slice(0, 3);
    };

    const createExtractionPrompt = (question: string, keyConstraints: string[]): string => {
        return `URGENT: Extract ONLY factual data that matches these EXACT constraints from: "${question}"

      REQUIRED CONSTRAINT MATCHES:
      ${keyConstraints.map((c, i) => `${i + 1}. "${c}" - Find exact matches or equivalents`).join('\n')}

      EXTRACTION RULES:
      - Extract EXACT numbers, dates, names, locations that match constraints

      CRITICAL: Extract data in JSON format:
      {
        "constraintMatches": {
          "${keyConstraints[0] || 'key1'}": "found value or null",
          "${keyConstraints[1] || 'key2'}": "found value or null"
        },
        "entityName": "main entity if identified",
        "additionalContext": "relevant supporting details"
      }

      If no constraint matches found, return: {"constraintMatches": {}, "entityName": null}`;
    };
      

    // Main research loop
    while (researchState.currentDepth < maxDepth) {
  const timeElapsed = Date.now() - startTime;
  if (timeElapsed >= timeLimit) {
    console.log('Time limit reached for BrowseComp');
    break;
  }

  researchState.currentDepth++;
  console.log(`BrowseComp research depth: ${researchState.currentDepth}`);

  let searchTopic = '';
  
  if (researchState.currentDepth === 1) {
    // Start with the full question first!
    searchTopic = lastUserMessage;
    console.log(`First search: Using full original question`);
  } else if (researchState.subquestions.length > 0) {
    searchTopic = researchState.subquestions.shift()!;
    researchState.answeredSubquestions.push(searchTopic);
    console.log(`Solving subquestion: "${searchTopic}"`);
  } else if (!researchState.nextSearchTopic || isGeneric(researchState.nextSearchTopic)) {
    console.warn('Generating subquestions from LLM');
     const subqPrompt = `Generate SPECIFIC research subquestions to solve this BrowseComp identification problem:

ORIGINAL QUESTION: "${lastUserMessage}"

${researchState.findings.length > 0 ? `CURRENT FINDINGS:
${researchState.findings.slice(-3).map((f, i) => `${i+1}. ${f.text.substring(0, 150)}...`).join('\n')}

SUBQUESTION GENERATION RULES:
1. Each subquestion MUST preserve the most identifying constraints from above
2. Focus on NARROWING DOWN to the specific individual/entity described
3. Combine multiple constraint types to create targeted searches
4. Progress from broad identification to specific details` : `SUBQUESTION GENERATION RULES:
1. First subquestion should identify the main subject using descriptive + temporal constraints
2. Second subquestion should add geographical + relational constraints  
3. Third subquestion should focus on the specific detail being asked
4. Each must preserve the exact constraints that make this individual unique`}

Generate subquestions as a JSON array:`;

    try {
      const subqResponse = await generateText({
        model: customModel(reasoningModel.apiIdentifier, true),
        prompt: subqPrompt,
      });

      let initialSubs = [];
      try {
        initialSubs = JSON.parse(subqResponse.text);
      } catch {
        const arrayMatch = subqResponse.text.match(/\[(.*?)\]/s);
        if (arrayMatch) {
          try {
            initialSubs = JSON.parse(arrayMatch[0]);
          } catch {
            const quotes = subqResponse.text.match(/"([^"]+)"/g);
            if (quotes) {
              initialSubs = quotes.map(q => q.replace(/"/g, ''));
            }
          }
        }
      }

      initialSubs = initialSubs
        .filter(q => typeof q === 'string' && q.length > 10 && q.length < 100)
        .slice(0, 8);

      console.log(`Generated ${initialSubs.length} subquestions:`, initialSubs);
      
      if (initialSubs.length > 0) {
        researchState.subquestions.push(...initialSubs);
        searchTopic = researchState.subquestions.shift() || generateFallbackQuery(lastUserMessage);
      } else {
        searchTopic = generateFallbackQuery(lastUserMessage);
      }
    } catch (err) {
      console.warn('Failed to generate subquestions:', err.message);
      searchTopic = extractKeyTerms(lastUserMessage);
    }
  } else {
    searchTopic = researchState.nextSearchTopic;
  }

  if (researchState.failedAttempts >= 2 && researchState.findings.length === 0) {
    console.warn('All hops failed — switching to keyword fallback mode');
    searchTopic = generateFallbackQuery(lastUserMessage);
  }

  console.log(`[Search Depth ${researchState.currentDepth}] Query: "${searchTopic}"`);
  
  
const searchQueries = Array(5).fill(searchTopic);

const allUrls = [];
for (const query of searchQueries) {
  try {
    const searchResult = await searchWithRetry(query);
    if (searchResult.success) {
      searchResult.data.forEach(result => {
        const existing = researchState.urlFrequencyMap.get(result.url);
        if (existing) {
          existing.frequency += 1;
        } else {
          researchState.urlFrequencyMap.set(result.url, {
            url: result.url,
            frequency: 1,
            title: result.title
          });
        }
        allUrls.push(result.url);
      });
    }
    await new Promise(resolve => setTimeout(resolve, 1000)); // Small delay
  } catch (error) {
    console.warn(`Search failed for: ${query}`);
  }
}


// Get most frequent URLs
const topUrls = getUrlsToProcess(researchState.urlFrequencyMap, researchState.processedUrls);

if (topUrls.length === 0) {
  console.log('No new URLs found across searches');
  researchState.failedAttempts++;
  if (researchState.failedAttempts >= researchState.maxFailedAttempts) {
    break;
  }
  continue;
}

console.log(`Found ${topUrls.length} most frequent URLs to process`);
      topUrls.forEach(url => researchState.processedUrls.add(url));
      
      if (topUrls.length === 0) {
        console.log('No new URLs to process, breaking loop');
        break;
      }

      

      const keyConstraints = await extractConstraintsWithLLM(lastUserMessage, reasoningModel);
      const extractionPrompt = createExtractionPrompt(lastUserMessage, keyConstraints); 
      const newFindings = await extractFromUrls(topUrls, extractionPrompt);
      researchState.findings.push(...newFindings);

      console.log(`Total findings so far: ${researchState.findings.length}`);

      console.log('Latest findings:');
      newFindings.forEach((finding, i) => {
        console.log(`${i+1}. [${finding.source}]: ${finding.text}`);
      });

      const analysis = await analyzeAndPlan(researchState.findings, searchTopic, lastUserMessage, researchState.subAnswers);

      if (!analysis) {
        researchState.failedAttempts++;
        if (researchState.failedAttempts >= researchState.maxFailedAttempts) {
          break;
        }
        continue;
      }
      // ADD THIS SECTION - Track subquestion answers
      if (analysis.subAnswer && searchTopic) {
        researchState.subAnswers.push({
          query: searchTopic,
          answer: analysis.subAnswer
        });
      }

      // Store subquestions and answers
      if (analysis.subquestions?.length > 0) {
        const newSubs = analysis.subquestions.filter(
          q => !researchState.subquestions.includes(q) && !researchState.answeredSubquestions.includes(q)
        );
        researchState.subquestions.push(...newSubs);
      }

      

      researchState.summaries.push(analysis.summary);
      console.log(`Analysis: ${analysis.summary}`);

      const shouldStop = (
        // Strong evidence with good confidence
        (analysis.hasAnswer && analysis.confidence === "high" && researchState.findings.length >= 3) ||
        
        // Medium confidence with substantial evidence
        (analysis.hasAnswer && analysis.confidence === "medium" && researchState.findings.length >= 6) ||
        
        // Lots of evidence even with lower confidence
        (researchState.findings.length >= 8)
      );

      if (shouldStop) {
        console.log(`Stopping research: hasAnswer=${analysis.hasAnswer}, confidence=${analysis.confidence}, findings=${researchState.findings.length}`);
        break;
      }

      if (!analysis.shouldContinue) {
        console.log('Analysis indicates to stop research');
        break;
      }

      researchState.nextSearchTopic = analysis.nextSearchTopic || '';

      if (researchState.currentDepth < maxDepth - 1) {
        console.log('Pausing 2 seconds between iterations...');
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }

    // Final synthesis
    console.log('Starting BrowseComp final synthesis');

    const keyConstraints = await extractConstraintsWithLLM(lastUserMessage, reasoningModel);

    const formattedSubAnswers = researchState.subAnswers.map(
      (a, i) => `${i + 1}. Q: ${a.query}\nA: ${a.answer}`
    ).join('\n');

    const frequencyReport = Array.from(researchState.urlFrequencyMap.entries())
          .sort((a, b) => b[1].frequency - a[1].frequency)
          .slice(0, 5)
          .map(([url, info], i) => `${i + 1}. ${url} (appeared ${info.frequency} times)`)
          .join('\n');
    
    const constraintCoverage = keyConstraints.map(constraint => {
    const matchingFindings = researchState.findings.filter(f => 
      f.text.toLowerCase().includes(constraint.toLowerCase())
    );
    return {
      constraint,
      coverage: matchingFindings.length,
      sources: matchingFindings.map(f => f.source)
    };
  });

  console.log('Constraint coverage analysis:');
  constraintCoverage.forEach(cc => {
    console.log(`  "${cc.constraint}": ${cc.coverage}/${researchState.findings.length} sources`);
  });

  const synthesisPrompt = `Answer this BrowseComp question: "${lastUserMessage}"

KEY IDENTIFYING CONSTRAINTS:
${keyConstraints.map((c, i) => `${i + 1}. "${c}"`).join('\n')}

CONSTRAINT COVERAGE ANALYSIS:
${constraintCoverage.map(cc => 
  `• "${cc.constraint}": found in ${cc.coverage}/${researchState.findings.length} sources`
).join('\n')}

RESEARCH FINDINGS (${researchState.findings.length} sources):
${researchState.findings.map((f,i) => `--- SOURCE ${i+1} (${f.source}) ---\n${f.text}`).join('\n\n')}

ANALYSIS STRATEGY:
1. For each potential answer, count how many KEY CONSTRAINTS it satisfies
2. Calculate match percentage: (matched constraints / total constraints) × 100
3. Prioritize answers with highest constraint match percentage
4. Use source frequency and quality as tiebreakers
5. Only use "Unknown" if no answer matches >40% of key constraints

CONSTRAINT SCORING EXAMPLES:
- Answer matches 4/5 constraints = 80% confidence
- Answer matches 2/6 constraints = 33% confidence  
- Perfect matches of specific numbers/names worth more than partial matches

The answer with the HIGHEST constraint match percentage is most likely correct.

FORMAT:
Explanation: [Which constraints were matched, what percentage, and why this answer scored highest]
Exact Answer: [The answer with best constraint coverage]
Confidence: [Percentage based on constraint matching and source quality]`;

  const finalAnalysis = await generateText({
    model: customModel(reasoningModel.apiIdentifier, true),
    maxTokens: 800,
    prompt: synthesisPrompt,
  });

    console.log('Raw final synthesis output:\n', finalAnalysis.text);

    const formattedResult = ensureValidBrowseCompFormat(finalAnalysis.text, lastUserMessage);
    console.log('BrowseComp research completed successfully');

    if (!formattedResult.includes('Exact Answer:')) {
      console.warn('Final output malformed or incomplete. Forcing fallback.');
      return new Response(JSON.stringify({
        content: `Explanation: The research did not produce a specific match for the requested constraints.
Exact Answer: Unknown
Confidence: 0%`
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    
    return new Response(JSON.stringify({ 
      content: formattedResult 
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
    
  } catch (error) {
    console.error('BrowseComp research error:', error);
    return new Response(JSON.stringify({ 
      content: ensureValidBrowseCompFormat('', lastUserMessage) 
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }
};

// Deep research implementation
const createDeepResearchTool = (reasoningModel: any, dataStream: any) => ({
  description: 'Perform deep research on a topic using an AI agent that coordinates search, extract, and analysis tools with reasoning steps.',
  parameters: z.object({
    topic: z.string().describe('The topic or question to research'),
    maxDepth: z.number().optional().describe('Maximum research depth (default 7)'),
  }),
  execute: async ({ topic, maxDepth = 7}) => {
    const startTime = Date.now();
    const timeLimit = 3.5 * 60 * 1000; 

    const researchState: ResearchState = {
      findings: [],
      summaries: [],
      nextSearchTopic: '',
      urlToSearch: '',
      currentDepth: 0,
      failedAttempts: 0,
      maxFailedAttempts: 3,
      completedSteps: 0,
      totalExpectedSteps: maxDepth * 5,
      subAnswers: [],
      urlFrequencyMap: new Map(),
    };

    // Initialize progress tracking
    dataStream.writeData({
      type: 'progress-init',
      content: {
        maxDepth,
        totalSteps: researchState.totalExpectedSteps,
      },
    });

    const addSource = (source: { url: string; title: string; description: string; }) => {
      dataStream.writeData({
        type: 'source-delta',
        content: source,
      });
    };

    const addActivity = (activity: {
      type: 'search' | 'extract' | 'analyze' | 'reasoning' | 'synthesis' | 'thought';
      status: 'pending' | 'complete' | 'error';
      message: string;
      timestamp: string;
      depth: number;
    }) => {
      if (activity.status === 'complete') {
        researchState.completedSteps++;
      }

      dataStream.writeData({
        type: 'activity-delta',
        content: {
          ...activity,
          depth: researchState.currentDepth,
          completedSteps: researchState.completedSteps,
          totalSteps: researchState.totalExpectedSteps,
        },
      });
    };

    const analyzeAndPlan = createAnalyzeAndPlan(reasoningModel, startTime, timeLimit);

    const extractFromUrlsWithActivity = async (urls: string[]) => {
      const filteredUrls = filterUrls(urls);

      const extractPromises = filteredUrls.map(async (url) => {
        try {
          addActivity({
            type: 'extract',
            status: 'pending',
            message: `Analyzing ${new URL(url).hostname}`,
            timestamp: new Date().toISOString(),
            depth: researchState.currentDepth,
          });

          const result = await app.extract([url], {
            prompt: `Extract key information about ${topic}. Focus on facts, data, and expert opinions. Analysis should be full of details and very comprehensive.`,
          });

          if (result.success) {
            addActivity({
              type: 'extract',
              status: 'complete',
              message: `Extracted from ${new URL(url).hostname}`,
              timestamp: new Date().toISOString(),
              depth: researchState.currentDepth,
            });

            if (Array.isArray(result.data)) {
              return result.data.map((item) => ({
                text: item.data,
                source: url,
              }));
            }
            return [{ text: result.data, source: url }];
          }
          return [];
        } catch {
          return [];
        }
      });

      const results = await Promise.all(extractPromises);
      return results.flat();
    };

    try {
      while (researchState.currentDepth < maxDepth) {
        const timeElapsed = Date.now() - startTime;
        if (timeElapsed >= timeLimit) {
          break;
        }

        researchState.currentDepth++;

        dataStream.writeData({
          type: 'depth-delta',
          content: {
            current: researchState.currentDepth,
            max: maxDepth,
            completedSteps: researchState.completedSteps,
            totalSteps: researchState.totalExpectedSteps,
          },
        });

        // Search phase
        addActivity({
          type: 'search',
          status: 'pending',
          message: `Searching for "${topic}"`,
          timestamp: new Date().toISOString(),
          depth: researchState.currentDepth,
        });

        let searchTopic = researchState.nextSearchTopic || topic;
        const searchResult = await searchWithRetry(searchTopic);

        if (!searchResult.success) {
          addActivity({
            type: 'search',
            status: 'error',
            message: `Search failed for "${searchTopic}"`,
            timestamp: new Date().toISOString(),
            depth: researchState.currentDepth,
          });

          researchState.failedAttempts++;
          if (researchState.failedAttempts >= researchState.maxFailedAttempts) {
            break;
          }
          continue;
        }

        addActivity({
          type: 'search',
          status: 'complete',
          message: `Found ${searchResult.data.length} relevant results`,
          timestamp: new Date().toISOString(),
          depth: researchState.currentDepth,
        });

        // Add sources from search results
        searchResult.data.forEach((result: any) => {
          addSource({
            url: result.url,
            title: result.title,
            description: result.description,
          });
        });

        // Extract phase
        const topUrls = searchResult.data.slice(0, 3).map((result: any) => result.url);

        const newFindings = await extractFromUrlsWithActivity([
          researchState.urlToSearch,
          ...topUrls,
        ]);
        researchState.findings.push(...newFindings);

        // Analysis phase
        addActivity({
          type: 'analyze',
          status: 'pending',
          message: 'Analyzing findings',
          timestamp: new Date().toISOString(),
          depth: researchState.currentDepth,
        });

        const analysis = await analyzeAndPlan(researchState.findings, searchTopic, topic, researchState.subAnswers);
        researchState.nextSearchTopic = analysis?.nextSearchTopic || '';
        researchState.urlToSearch = analysis?.urlToSearch || '';
        researchState.summaries.push(analysis?.summary || '');
        
        if (analysis.subAnswer) {
          researchState.subAnswers.push({
            query: analysis.lastQuery || searchTopic,
            answer: analysis.subAnswer
          });
        }

        if (!analysis) {
          addActivity({
            type: 'analyze',
            status: 'error',
            message: 'Failed to analyze findings',
            timestamp: new Date().toISOString(),
            depth: researchState.currentDepth,
          });

          researchState.failedAttempts++;
          if (researchState.failedAttempts >= researchState.maxFailedAttempts) {
            break;
          }
          continue;
        }

        addActivity({
          type: 'analyze',
          status: 'complete',
          message: analysis.summary,
          timestamp: new Date().toISOString(),
          depth: researchState.currentDepth,
        });

        if (!analysis.shouldContinue || analysis.gaps.length === 0) {
          break;
        }

        topic = analysis.gaps.shift() || topic;
      }

      // Final synthesis
      addActivity({
        type: 'synthesis',
        status: 'pending',
        message: 'Preparing final analysis',
        timestamp: new Date().toISOString(),
        depth: researchState.currentDepth,
      });

      const finalAnalysis = await generateText({
        model: customModel(reasoningModel.apiIdentifier, true),
        maxTokens: 16000,
        prompt: `Create a comprehensive long analysis of ${topic} based on these findings:
${researchState.findings.map((f) => `[From ${f.source}]: ${f.text}`).join('\n')}
${researchState.summaries.map((s) => `[Summary]: ${s}`).join('\n')}
Provide all the thoughts processes including findings details,key insights, conclusions, and any remaining uncertainties. Include citations to sources where appropriate. This analysis should be very comprehensive and full of details. It is expected to be very long, detailed and comprehensive.`,
      });

      console.log('Final synthesis raw output:\n', finalAnalysis.text);

      addActivity({
        type: 'synthesis',
        status: 'complete',
        message: 'Research completed',
        timestamp: new Date().toISOString(),
        depth: researchState.currentDepth,
      });

      dataStream.writeData({
        type: 'finish',
        content: finalAnalysis.text,
      });

      return {
        success: true,
        data: {
          findings: researchState.findings,
          analysis: finalAnalysis.text,
          completedSteps: researchState.completedSteps,
          totalSteps: researchState.totalExpectedSteps,
        },
      };
    } catch (error: any) {
      console.error('Deep research error:', error);

      addActivity({
        type: 'thought',
        status: 'error',
        message: `Research failed: ${error.message}`,
        timestamp: new Date().toISOString(),
        depth: researchState.currentDepth,
      });

      return {
        success: false,
        error: error.message,
        data: {
          findings: researchState.findings,
          completedSteps: researchState.completedSteps,
          totalSteps: researchState.totalExpectedSteps,
        },
      };
    }
  },
});

// Main POST handler
export async function POST(request: Request) {
  try {
    const maxDuration = process.env.MAX_DURATION ? parseInt(process.env.MAX_DURATION) : 300;
    
    const {
      id,
      messages,
      modelId,
      reasoningModelId,
      experimental_deepResearch = false,
    } = await request.json();

    // Check if this is a BrowseComp request
    const userAgent = request.headers.get("user-agent") || "";
    const isSimpleEval = userAgent.includes("python-requests");
    
    if (isSimpleEval && messages && messages.length > 0) {
      return await handleBrowseCompRequest(messages, modelId, reasoningModelId);
    }

    // Normal authentication flow for non-BrowseComp requests
    const session = await ensureAuthenticatedSession();

    // Apply rate limiting
    const identifier = session.user.id;
    const { success, limit, reset, remaining } = await rateLimiter.limit(identifier);

    if (!success) {
      return new Response(`Too many requests`, { status: 429 });
    }

    const model = models.find((model) => model.id === modelId);
    const reasoningModel = reasoningModels.find((model) => model.id === reasoningModelId);

    if (!model || !reasoningModel) {
      return new Response('Model not found', { status: 404 });
    }

    const coreMessages = convertToCoreMessages(messages);
    const userMessage = getMostRecentUserMessage(coreMessages);

    if (!userMessage) {
      return new Response('No user message found', { status: 400 });
    }

    const chat = await getChatById({ id });

    if (!chat) {
      const title = await generateTitleFromUserMessage({ message: userMessage });
      await saveChat({ id, userId: session.user.id, title });
    }

    const userMessageId = generateUUID();

    await saveMessages({
      messages: [
        { ...userMessage, id: userMessageId, createdAt: new Date(), chatId: id },
      ],
    });

    return createDataStreamResponse({
      execute: (dataStream) => {
        dataStream.writeData({
          type: 'user-message-id',
          content: userMessageId,
        });

        const result = streamText({
          model: customModel(model.apiIdentifier, false),
          system: systemPrompt,
          messages: coreMessages,
          maxSteps: 10,
          experimental_activeTools: experimental_deepResearch ? allTools : firecrawlTools,
          tools: {
            search: {
              description: "Search for web pages. Normally you should call the extract tool after this one to get a specific data point if search doesn't the exact data you need.",
              parameters: z.object({
                query: z.string().describe('Search query to find relevant web pages'),
                maxResults: z.number().optional().describe('Maximum number of results to return (default 10)'),
              }),
              execute: async ({ query, maxResults = 3 }) => {
                try {
                  const searchResult = await app.search(query);

                  if (!searchResult.success) {
                    return {
                      error: `Search failed: ${searchResult.error}`,
                      success: false,
                    };
                  }

                  const resultsWithFavicons = searchResult.data.map((result: any) => {
                    const url = new URL(result.url);
                    const favicon = `https://www.google.com/s2/favicons?domain=${url.hostname}&sz=32`;
                    return {
                      ...result,
                      favicon
                    };
                  });

                  searchResult.data = resultsWithFavicons;

                  return {
                    data: searchResult.data,
                    success: true,
                  };
                } catch (error: any) {
                  return {
                    error: `Search failed: ${error.message}`,
                    success: false,
                  };
                }
              },
            },
            extract: {
              description: 'Extract structured data from web pages. Use this to get whatever data you need from a URL. Any time someone needs to gather data from something, use this tool.',
              parameters: z.object({
                urls: z.array(z.string()).describe('Array of URLs to extract data from'),
                prompt: z.string().describe('Description of what data to extract'),
              }),
              execute: async ({ urls, prompt }) => {
                try {
                  const scrapeResult = await app.extract(urls, {
                    prompt,
                  });

                  if (!scrapeResult.success) {
                    return {
                      error: `Failed to extract data: ${scrapeResult.error}`,
                      success: false,
                    };
                  }

                  return {
                    data: scrapeResult.data,
                    success: true,
                  };
                } catch (error: any) {
                  console.error('Extraction error:', error);
                  return {
                    error: `Extraction failed: ${error.message}`,
                    success: false,
                  };
                }
              },
            },
            scrape: {
              description: 'Scrape web pages. Use this to get from a page when you have the url.',
              parameters: z.object({
                url: z.string().describe('URL to scrape'),
              }),
              execute: async ({ url }: { url: string }) => {
                try {
                  const scrapeResult = await app.scrapeUrl(url);

                  if (!scrapeResult.success) {
                    return {
                      error: `Failed to extract data: ${scrapeResult.error}`,
                      success: false,
                    };
                  }

                  return {
                    data: scrapeResult.markdown ?? 'Could get the page content, try using search or extract',
                    success: true,
                  };
                } catch (error: any) {
                  console.error('Extraction error:', error);
                  return {
                    error: `Extraction failed: ${error.message}`,
                    success: false,
                  };
                }
              },
            },
            deepResearch: createDeepResearchTool(reasoningModel, dataStream),
          },
          onFinish: async ({ response }) => {
            if (session.user?.id) {
              try {
                const responseMessagesWithoutIncompleteToolCalls = sanitizeResponseMessages(response.messages);

                await saveMessages({
                  messages: responseMessagesWithoutIncompleteToolCalls.map((message) => {
                    const messageId = generateUUID();

                    if (message.role === 'assistant') {
                      dataStream.writeMessageAnnotation({
                        messageIdFromServer: messageId,
                      });
                    }

                    return {
                      id: messageId,
                      chatId: id,
                      role: message.role,
                      content: message.content,
                      createdAt: new Date(),
                    };
                  }),
                });
              } catch (error) {
                console.error('Failed to save chat');
              }
            }
          },
          experimental_telemetry: {
            isEnabled: true,
            functionId: 'stream-text',
          },
        });

        result.mergeIntoDataStream(dataStream);
      },
    });

  } catch (error) {
    console.error('POST handler error:', error);
    const userAgent = request.headers.get("user-agent") || "";
    if (userAgent.includes("python-requests")) {
      return new Response(JSON.stringify({ 
        content: `Explanation: An error occurred while processing your request.
Exact Answer: Unknown
Confidence: 0%`
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    
    return new Response('Internal Server Error', { status: 500 });
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');

  if (!id) {
    return new Response('Not Found', { status: 404 });
  }

  try {
    const session = await ensureAuthenticatedSession();

    const chat = await getChatById({ id });

    if (chat.userId !== session.user.id) {
      return new Response('Unauthorized', { status: 401 });
    }

    await deleteChatById({ id });

    return new Response('Chat deleted', { status: 200 });
  } catch (error) {
    return new Response('An error occurred while processing your request', {
      status: 500,
    });
  }
}


